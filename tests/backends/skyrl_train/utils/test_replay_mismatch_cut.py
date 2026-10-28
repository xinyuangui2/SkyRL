import sys
import types
from enum import Enum

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    build_token_metadata_layout,
)
from skyrl.backends.skyrl_train.utils import replay_utils
from skyrl.backends.skyrl_train.utils.replay_utils import (
    ReplayMismatchCut,
    ReplayMismatchFlags,
    replay_mismatch_flags_to_batch,
)


@pytest.fixture
def parallel_state(monkeypatch):
    try:
        import megatron.core.parallel_state as mpu
    except ModuleNotFoundError:
        megatron = types.ModuleType("megatron")
        core = types.ModuleType("megatron.core")
        mpu = types.ModuleType("megatron.core.parallel_state")
        megatron.core = core
        core.parallel_state = mpu
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", core)
        monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", mpu)

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    return mpu


def test_flags_tokens_with_low_overlap_at_any_layer():
    flags = ReplayMismatchFlags(max_overlap=1, local_valid=torch.tensor([True, True, True, False]))
    natural = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
    # Overlaps 3, 1, 2, 0; the last token is padding and is never flagged.
    flags.observe(torch.tensor([[2, 1, 0], [1, 5, 6], [0, 2, 7], [5, 6, 7]]), natural)
    assert flags.flagged.tolist() == [False, True, False, False]
    # A later layer flags token 0; earlier flags stick.
    flags.observe(torch.tensor([[5, 6, 7], [0, 1, 2], [0, 2, 7], [5, 6, 7]]), natural)
    assert flags.flagged.tolist() == [True, True, False, False]
    assert flags.flagged_token_layers.item() == 2
    assert flags.valid_token_layers.item() == 6
    assert flags.overlap_sum.item() == (3 + 1 + 2) + (0 + 3 + 2)
    assert flags.overlap_min.item() == 0


def test_observe_rejects_token_count_mismatch():
    flags = ReplayMismatchFlags(max_overlap=0, local_valid=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="Router saw 2 tokens"):
        flags.observe(torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long))


def test_cut_grad_zeroes_flagged_tokens_in_sequence_batch_order():
    # Router order is [sequence, batch] flattened: token index = s * batch + b.
    flags = ReplayMismatchFlags(max_overlap=0, local_valid=torch.ones(6, dtype=torch.bool))
    flags.flagged = torch.tensor([False, True, False, False, True, False])
    grad = torch.ones(3, 2, 4)
    cut = flags.cut_grad(grad)
    assert cut[0, 1].abs().sum() == 0 and cut[2, 0].abs().sum() == 0
    assert cut.sum() == 4 * 4


def test_flags_map_to_next_token_loss_positions_unpacked(parallel_state):
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
    layout = build_token_metadata_layout(attention_mask, attention_mask.device, packed=False, fp8_enabled=False)
    seq_len = layout.aligned_sequence_length
    flags = ReplayMismatchFlags(max_overlap=0, local_valid=torch.ones(seq_len * 2, dtype=torch.bool))
    # Flag row 0's second real token (batch position 2) and row 1's first token (batch position 0).
    model_flags = torch.zeros(2, seq_len, dtype=torch.bool)
    model_flags[0, 1] = True
    model_flags[1, 0] = True
    flags.flagged = model_flags.t().reshape(-1)
    batch_flags = replay_mismatch_flags_to_batch(flags, layout)
    expected = torch.zeros(2, 3, dtype=torch.bool)
    expected[0, 2] = True
    expected[1, 0] = True
    assert torch.equal(batch_flags, expected)


def test_flags_map_to_next_token_loss_positions_packed(parallel_state):
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
    layout = build_token_metadata_layout(attention_mask, attention_mask.device, packed=True, fp8_enabled=False)
    num_tokens = layout.aligned_sequence_length
    padded = layout.padded_sequence_lengths
    flags = ReplayMismatchFlags(max_overlap=0, local_valid=torch.ones(num_tokens, dtype=torch.bool))
    packed_flags = torch.zeros(num_tokens, dtype=torch.bool)
    packed_flags[1] = True  # row 0, second real token -> batch position 2
    packed_flags[padded[0]] = True  # row 1, first token -> batch position 0
    flags.flagged = packed_flags
    batch_flags = replay_mismatch_flags_to_batch(flags, layout)
    expected = torch.zeros(2, 3, dtype=torch.bool)
    expected[0, 2] = True
    expected[1, 0] = True
    assert torch.equal(batch_flags, expected)


class _ResidualLayer(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, hidden)

    def forward(self, hidden_states):
        # Mix tokens so a flagged token's gradient would otherwise reach the others' inputs and back.
        return hidden_states + self.linear(hidden_states.mean(dim=0, keepdim=True)).expand_as(hidden_states)


@pytest.fixture
def router_replay(monkeypatch):
    class RouterReplayAction(Enum):
        REPLAY_FORWARD = "replay_forward"
        REPLAY_BACKWARD = "replay_backward"

    class RouterReplay:
        def __init__(self, target):
            self.target_topk_idx = target
            self.router_replay_action = RouterReplayAction.REPLAY_FORWARD

        def get_replay_topk(self, scores, topk, num_groups=None, group_topk=None, default_compute_topk=None):
            return scores.gather(1, self.target_topk_idx), self.target_topk_idx

    module = types.ModuleType("megatron.core.transformer.moe.router_replay")
    module.RouterReplay = RouterReplay
    module.RouterReplayAction = RouterReplayAction
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.router_replay", module)
    return RouterReplay


def test_cut_removes_flagged_token_gradient_end_to_end(monkeypatch, router_replay):
    hidden, seq_len = 4, 3
    layers = torch.nn.ModuleList([_ResidualLayer(hidden), _ResidualLayer(hidden)])
    monkeypatch.setattr(replay_utils, "_decoder_layers", lambda chunk: list(chunk))

    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]] * seq_len)
    # Token 1's replayed experts {2, 3} miss its natural top-2 {0, 1}.
    replayed = torch.tensor([[0, 1], [2, 3], [1, 0]])
    router = router_replay(replayed)

    def natural_topk(scores, topk, num_groups=None, group_topk=None):
        return torch.topk(scores, topk, dim=1)

    x = torch.randn(seq_len, 1, hidden, requires_grad=True)
    with ReplayMismatchCut(0, [layers]) as cut:
        cut.start_microbatch(torch.ones(seq_len, dtype=torch.bool))
        router.get_replay_topk(scores, 2, None, None, natural_topk)
        out = x
        for layer in layers:
            out = layer(out)
        out.sum().backward()
    assert cut.current is None
    assert x.grad[1].abs().sum() == 0
    assert x.grad[0].abs().sum() > 0 and x.grad[2].abs().sum() > 0

    # Outside the schedule nothing is flagged or cut (a fresh input: the hook stays on the old one).
    x = x.detach().requires_grad_()
    router.get_replay_topk(scores, 2, None, None, natural_topk)
    out = x
    for layer in layers:
        out = layer(out)
    out.sum().backward()
    assert x.grad[1].abs().sum() > 0
