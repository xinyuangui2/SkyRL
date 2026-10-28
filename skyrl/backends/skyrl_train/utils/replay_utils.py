"""
Utility functions for MoE Router Replay.
"""

from contextlib import contextmanager

import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
    align_packed_token_metadata,
    align_token_metadata,
    scatter_packed_token_values_to_batch,
)
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor


def replay_padding_row(
    topk: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Return one dummy route of ``topk`` distinct experts.

    Synthetic padding tokens still have to satisfy Megatron's dropless
    ``tokens * topk`` dispatcher invariant, so they route to distinct experts
    rather than repeating one. ``router_padding_mask`` excludes these rows from
    expert-bias accounting.
    """
    if topk < 1:
        raise ValueError(f"Replay route padding requires a positive topk dimension, got {topk}")
    return torch.arange(topk, dtype=dtype, device=device)


def make_replay_padding_indices(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Return dummy routes with ``topk`` distinct experts in every row."""
    if not shape:
        raise ValueError(f"Replay route padding requires a positive topk dimension, got {shape}")
    padding_row = replay_padding_row(shape[-1], dtype=dtype, device=device)
    return padding_row.expand(shape).clone()


def patch_topk_router_layer_number():
    """Monkey-patch TopKRouter.set_layer_number to propagate the global layer
    number to the RouterReplay instance.

    DeepSeek V3 (and similar) architectures have dense FFN layers before the MoE
    layers.  vLLM reports routing indices for ALL transformer layers (including
    dense), but Megatron only creates RouterReplay instances for MoE layers.
    Storing the global layer_number on each RouterReplay instance lets us map
    vLLM's per-layer data to the correct MoE router even when dense layers are
    present.

    Must be called BEFORE model creation (i.e. before make_megatron_module).
    """
    try:
        from megatron.core.transformer.moe.router import TopKRouter
    except ImportError:
        return

    if getattr(TopKRouter, "_set_layer_number_patched", False):
        return

    original_set_layer_number = TopKRouter.set_layer_number

    def patched_set_layer_number(self, layer_number: int):
        original_set_layer_number(self, layer_number)
        if self.router_replay is not None:
            self.router_replay.layer_number = layer_number

    TopKRouter.set_layer_number = patched_set_layer_number
    TopKRouter._set_layer_number_patched = True


def _split_replay_indices(rollout_expert_indices: torch.Tensor) -> list[torch.Tensor]:
    # Megatron flattens router logits in [sequence, batch] order.
    per_layer = rollout_expert_indices.permute(2, 1, 0, 3).contiguous().to(torch.int32)
    return list(per_layer.flatten(1, 2).unbind(0))


def scatter_router_padding_mask_for_model(
    router_padding_mask: torch.Tensor | None,
    model,
    model_config,
) -> torch.Tensor | None:
    """Match the mask layout to sequence-parallel hidden states at model entry."""
    if router_padding_mask is None or not model_config.sequence_parallel:
        return router_padding_mask

    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
    from megatron.core.utils import unwrap_model

    unwrapped_model = unwrap_model(model)
    # GPTModel scatters its mask beside the embedding on the first PP stage. HybridModel
    # scatters only the embedding, so its mask must always be scattered here.
    if not isinstance(unwrapped_model, HybridModel) and unwrapped_model.pre_process:
        return router_padding_mask
    return (
        scatter_to_sequence_parallel_region(router_padding_mask.transpose(0, 1).contiguous())
        .transpose(0, 1)
        .contiguous()
    )


def _get_current_pp_stage_layer_range(model_config) -> tuple[int, int]:
    """Return the current PP rank's transformer-layer range as (start_layer,
    num_layers).

    Prefer Megatron's own helpers so replay indexing stays aligned with the
    actual model partition, including embedding/loss pipeline accounting.
    """
    import megatron.core.parallel_state as mpu
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    offset = get_transformer_layer_offset(model_config, pp_rank=pp_rank)
    num_layers = get_num_layers_to_build(model_config, pp_rank=pp_rank)
    return offset, num_layers


def _get_local_router_layer_indices(model_config, global_num_layers: int, instances: list) -> list[int]:
    local_layer_offset, local_num_layers = _get_current_pp_stage_layer_range(model_config)
    if local_num_layers == len(instances):
        layer_indices = list(range(local_layer_offset, local_layer_offset + local_num_layers))
    else:
        # Dense-layer mismatch: map each MoE router to its global layer index.
        # Prefer the patched layer_number; fall back to offset-based mapping
        # (assumes dense layers precede MoE layers).
        layer_indices = []
        for local_router_index, router_instance in enumerate(instances):
            layer_number = getattr(router_instance, "layer_number", None)
            if layer_number is not None:
                layer_index = layer_number - 1
            else:
                layer_index = local_layer_offset + local_router_index + (local_num_layers - len(instances))
            layer_indices.append(layer_index)

    if any(layer_index < 0 or layer_index >= global_num_layers for layer_index in layer_indices):
        raise ValueError(
            f"Router replay layer indices {layer_indices} out of range for data with {global_num_layers} layers"
        )
    return layer_indices


def setup_per_microbatch_replay_forward(
    rollout_expert_indices: PackedTensor,
    router_padding_mask: torch.Tensor | None,
    attention_mask: torch.Tensor,
    model,
    model_config,
    metadata_layout: TokenMetadataLayout,
    remove_microbatch_padding: bool = False,
) -> dict[str, torch.Tensor]:
    """Set up router replay and return its model-facing keyword arguments.

    Replay indices arrive packed to their real tokens (``[sum(seqlen), layers, topk]`` plus
    ``cu_seqlens``) while the router padding mask arrives batch-major; both undergo matching
    padding removal or packing and CP sharding against the shared layout. Their destinations
    then differ: indices are TP-sliced and installed into per-layer ``RouterReplay``
    instances, while the mask follows Megatron's model-specific sequence-parallel
    path and is passed to the model as ``padding_mask``.

    Handles context parallelism: when CP > 1, the sequence is split into
    2*cp_size chunks with each CP rank receiving a front chunk and a back
    chunk (for causal-mask load balancing). Replay indices are split using
    the same pattern so they stay aligned with the tokens each rank sees.

    Handles sequence parallelism: when TP > 1, the sequence is split across
    TP ranks, so each rank's MoE router only sees its local chunk of tokens.

    Handles dense-layer mismatch: DeepSeek V3-style models have dense FFN
    layers before the MoE layers. vLLM reports routing indices for ALL
    transformer layers, but Megatron only has RouterReplay instances for MoE
    layers. We use each instance's global layer_number (set by the patched
    TopKRouter.set_layer_number) to index into the correct slice of the data.

    Handles pipeline parallelism: when PP > 1, transformer layers are split
    across PP ranks, so each rank only sees its local RouterReplay instances. In cases
    where the number of local RouterReplay instances does not match the local
    layer count, indicating that the model has dense layers before MoE layers,
    we use the global layer_number to index into the correct slice of the data.

    """
    import megatron.core.parallel_state as mpu
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    if router_padding_mask is None:
        raise ValueError("router_padding_mask is required with rollout_expert_indices")
    if len(rollout_expert_indices.row_shape) != 2:
        raise ValueError(f"Expected [tokens, layers, topk] replay indices, got {rollout_expert_indices!r}")

    if router_padding_mask.shape != attention_mask.shape:
        raise ValueError(
            f"router_padding_mask shape {router_padding_mask.shape} does not match "
            f"attention_mask shape {attention_mask.shape}"
        )
    if router_padding_mask.device != rollout_expert_indices.device:
        raise ValueError("rollout_expert_indices and router_padding_mask must be on the same device")

    num_captured_layers, topk = rollout_expert_indices.row_shape
    instances = RouterReplay.global_router_replay_instances
    local_layer_indices = _get_local_router_layer_indices(
        model_config,
        num_captured_layers,
        instances,
    )
    layer_index = torch.tensor(local_layer_indices, dtype=torch.long, device=rollout_expert_indices.device)
    local_rollout_expert_indices = PackedTensor(
        rollout_expert_indices.values.index_select(1, layer_index),
        rollout_expert_indices.cu_seqlens,
    )

    if (metadata_layout.padded_sequence_lengths is not None) != remove_microbatch_padding:
        raise ValueError("Shared token metadata layout does not match the model packing mode")
    aligned_router_padding_mask = align_token_metadata(router_padding_mask.to(torch.bool), metadata_layout, True)
    route_padding = replay_padding_row(
        topk,
        dtype=rollout_expert_indices.dtype,
        device=rollout_expert_indices.device,
    )
    aligned_rollout_expert_indices = align_packed_token_metadata(
        local_rollout_expert_indices,
        metadata_layout,
        route_padding,
    )

    # TP splitting: sequence parallelism across the tensor model parallel region
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size > 1:
        tp_rank = mpu.get_tensor_model_parallel_rank()
        seq_len = aligned_rollout_expert_indices.shape[1]
        chunk_size = seq_len // tp_size
        aligned_rollout_expert_indices = aligned_rollout_expert_indices[
            :, tp_rank * chunk_size : (tp_rank + 1) * chunk_size, :, :
        ]
    RouterReplay.set_replay_data(_split_replay_indices(aligned_rollout_expert_indices))
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    model_router_padding_mask = scatter_router_padding_mask_for_model(
        aligned_router_padding_mask,
        model,
        model_config,
    )
    return {"padding_mask": model_router_padding_mask}


def setup_per_microbatch_replay_backward() -> None:
    """Switch RouterReplay to backward mode so that activation-checkpoint
    recomputation during the backward pass consumes indices from
    ``replay_backward_list`` in FIFO order (populated during the forward pass).
    """
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)


def clear_router_replay():
    """Clear all router replay state."""
    from megatron.core.transformer.moe.router_replay import RouterReplay

    RouterReplay.clear_global_indices()
    RouterReplay.clear_global_router_replay_action()


@contextmanager
def router_replay_schedule(enabled: bool):
    """Isolate global RouterReplay state to one Megatron pipeline schedule.

    The backward FIFO spans all microbatches in a training schedule, so it must
    only be cleared at schedule boundaries. Forward-only schedules leave that
    FIFO unconsumed, and failed schedules may leave it partially consumed.
    """
    if not enabled:
        yield
        return

    clear_router_replay()
    try:
        yield
    finally:
        clear_router_replay()


class ReplayMismatchFlags:
    """One microbatch's tokens whose replayed experts disagree with their own router.

    A token is flagged when, at any MoE layer, its replayed top-k shares at most ``max_overlap``
    experts with the top-k its own router scores pick. Flags live in the router's local token order
    (``[sequence, batch]`` flattened, sequence-parallel shard). They are only complete once the whole
    forward has run, so the gradient hooks and the loss read them after the forward.
    """

    def __init__(self, max_overlap: int, local_valid: torch.Tensor):
        self.max_overlap = max_overlap
        # [local tokens] bool, False for padding tokens (their replayed routes are dummies).
        self.local_valid = local_valid
        self.flagged = torch.zeros_like(local_valid)
        self.flagged_token_layers = torch.zeros((), dtype=torch.long, device=local_valid.device)
        self.overlap_sum = torch.zeros((), dtype=torch.long, device=local_valid.device)
        self.valid_token_layers = torch.zeros((), dtype=torch.long, device=local_valid.device)
        self.overlap_min = torch.full((), torch.iinfo(torch.long).max, dtype=torch.long, device=local_valid.device)

    def observe(self, replayed_indices: torch.Tensor, natural_indices: torch.Tensor) -> None:
        """Record one MoE layer's replayed vs natural top-k (both ``[local tokens, topk]``)."""
        if replayed_indices.shape[0] != self.flagged.shape[0]:
            raise ValueError(
                f"Router saw {replayed_indices.shape[0]} tokens, but the replay mismatch mask holds "
                f"{self.flagged.shape[0]}"
            )
        replayed = replayed_indices.to(natural_indices.device, torch.long)
        overlap = (replayed[:, :, None] == natural_indices.long()[:, None, :]).any(-1).sum(-1)
        layer_flagged = (overlap <= self.max_overlap) & self.local_valid
        self.flagged |= layer_flagged
        self.flagged_token_layers += layer_flagged.sum()
        valid_overlap = overlap[self.local_valid]
        self.overlap_sum += valid_overlap.sum()
        self.valid_token_layers += valid_overlap.numel()
        if valid_overlap.numel():
            self.overlap_min = torch.minimum(self.overlap_min, valid_overlap.min())

    def cut_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Zero a ``[sequence, batch, ...]`` hidden-state gradient at flagged tokens."""
        seq_len, batch_size = grad.shape[0], grad.shape[1]
        if seq_len * batch_size != self.flagged.numel():
            raise ValueError(
                f"Hidden-state gradient {tuple(grad.shape)} does not match {self.flagged.numel()} routed tokens"
            )
        keep = (~self.flagged).view(seq_len, batch_size, *([1] * (grad.dim() - 2)))
        return grad * keep.to(grad.dtype)


def local_replay_token_mask(
    router_padding_mask: torch.Tensor,
    metadata_layout: TokenMetadataLayout,
) -> torch.Tensor:
    """Return this rank's real (non-padding) routed tokens in the router's local token order."""
    import megatron.core.parallel_state as mpu

    # The router padding mask is True at padding tokens; aligned the same way as the replay indices.
    aligned_padding = align_token_metadata(router_padding_mask.to(torch.bool), metadata_layout, True)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size > 1:
        tp_rank = mpu.get_tensor_model_parallel_rank()
        chunk_size = aligned_padding.shape[1] // tp_size
        aligned_padding = aligned_padding[:, tp_rank * chunk_size : (tp_rank + 1) * chunk_size]
    return (~aligned_padding).t().reshape(-1)


def replay_mismatch_flags_to_batch(
    flags: ReplayMismatchFlags,
    metadata_layout: TokenMetadataLayout,
) -> torch.Tensor:
    """Map the microbatch's flags to canonical ``[batch, seq_len - 1]`` next-token loss positions.

    The logit at a token's position scores the next token, so a flagged token drops that position.
    """
    import megatron.core.parallel_state as mpu

    if metadata_layout.padded_sequence_lengths is not None:
        batch_size = 1
    else:
        batch_size = metadata_layout.attention_mask.shape[0]
    local = flags.flagged.view(-1, batch_size)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size > 1:
        gathered = torch.empty((local.shape[0] * tp_size, batch_size), dtype=torch.uint8, device=local.device)
        torch.distributed.all_gather_into_tensor(
            gathered, local.to(torch.uint8).contiguous(), group=mpu.get_tensor_model_parallel_group()
        )
        local = gathered.to(torch.bool)
    model_flags = local.t().contiguous()  # [batch, aligned seq] (packed: [1, tokens on this CP rank])

    if metadata_layout.padded_sequence_lengths is not None:
        return scatter_packed_token_values_to_batch(model_flags, metadata_layout, False)
    if metadata_layout.context_parallel_size > 1:
        raise ValueError("Replay mismatch cut requires remove_microbatch_padding with context parallelism")
    attention_mask = metadata_layout.attention_mask
    batch_flags = torch.zeros(attention_mask.shape, dtype=torch.bool, device=model_flags.device)
    for row_index, sequence_length in enumerate(metadata_layout.sequence_lengths):
        batch_flags[row_index, attention_mask[row_index]] = model_flags[row_index, :sequence_length]
    return batch_flags[:, :-1]


def replay_mismatch_metrics(flags: ReplayMismatchFlags) -> dict[str, float]:
    """Sum the per-rank routing statistics over this microbatch's TP x CP shards."""
    import megatron.core.parallel_state as mpu

    group = mpu.get_tensor_and_context_parallel_group()
    sums = torch.stack([flags.flagged_token_layers, flags.overlap_sum, flags.valid_token_layers])
    overlap_min = flags.overlap_min.clone()
    if torch.distributed.is_initialized() and group.size() > 1:
        torch.distributed.all_reduce(sums, group=group)
        torch.distributed.all_reduce(overlap_min, op=torch.distributed.ReduceOp.MIN, group=group)
    flagged_token_layers, overlap_sum, valid_token_layers = sums.tolist()
    metrics = {"replay_mismatch/flagged_token_layers": float(flagged_token_layers)}
    if valid_token_layers:
        metrics["replay_mismatch/overlap_mean"] = overlap_sum / valid_token_layers
        metrics["replay_mismatch/overlap_min"] = float(overlap_min.item())
    return metrics


class ReplayMismatchCut:
    """Opt-in safeguard against routing-replay mismatch (``moe_replay_mismatch_cut_overlap``).

    With rollout routing replay, a token whose trainer-side hidden state has drifted from the
    rollout can be forced through experts its own router barely weights, and its gradient can be
    amplified enough to dominate the update. For every token flagged by ``ReplayMismatchFlags``,
    this zeroes the residual-stream gradient at every decoder-layer input and (in the loss) drops
    its loss position. Routing itself is unchanged: flagged tokens still use the replayed experts.

    Active for one training schedule. Requires PP=1, so each microbatch's forward and backward run
    back to back.
    """

    _active: "ReplayMismatchCut | None" = None

    def __init__(self, max_overlap: int, model_chunks: list):
        self.max_overlap = max_overlap
        self.model_chunks = model_chunks
        self.current: ReplayMismatchFlags | None = None
        self._handles = []

    def start_microbatch(self, local_valid: torch.Tensor) -> ReplayMismatchFlags:
        self.current = ReplayMismatchFlags(self.max_overlap, local_valid)
        return self.current

    def _cut_layer_input(self, module, args, kwargs):
        hidden_states = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        flags = self.current
        if flags is not None and torch.is_tensor(hidden_states) and hidden_states.requires_grad:
            hidden_states.register_hook(flags.cut_grad)

    def __enter__(self) -> "ReplayMismatchCut":
        _patch_router_replay_mismatch_observer()
        for chunk in self.model_chunks:
            for layer in _decoder_layers(chunk):
                self._handles.append(layer.register_forward_pre_hook(self._cut_layer_input, with_kwargs=True))
        ReplayMismatchCut._active = self
        return self

    def __exit__(self, *exc) -> None:
        ReplayMismatchCut._active = None
        self.current = None
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _decoder_layers(model) -> list:
    from megatron.core.utils import unwrap_model

    unwrapped = unwrap_model(model)
    for owner in (unwrapped, getattr(unwrapped, "language_model", None)):
        decoder = getattr(owner, "decoder", None)
        if decoder is not None and hasattr(decoder, "layers"):
            return list(decoder.layers)
    raise ValueError(f"Replay mismatch cut could not find decoder layers on {type(unwrapped).__name__}")


def _patch_router_replay_mismatch_observer() -> None:
    """Wrap ``RouterReplay.get_replay_topk`` so replayed forwards report their natural top-k."""
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    if getattr(RouterReplay, "_mismatch_observer_patched", False):
        return
    original_get_replay_topk = RouterReplay.get_replay_topk

    def get_replay_topk(self, scores, topk, num_groups=None, group_topk=None, default_compute_topk=None):
        result = original_get_replay_topk(self, scores, topk, num_groups, group_topk, default_compute_topk)
        cut = ReplayMismatchCut._active
        if (
            cut is not None
            and cut.current is not None
            and self.router_replay_action == RouterReplayAction.REPLAY_FORWARD
        ):
            with torch.no_grad():
                _, natural_indices = default_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                cut.current.observe(result[1], natural_indices)
        return result

    RouterReplay.get_replay_topk = get_replay_topk
    RouterReplay._mismatch_observer_patched = True
