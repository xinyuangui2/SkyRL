"""
CPU tests for SkyRLVLMGymGenerator.

The generator's skycap server is real; its renderer is skycap's test renderer (characters are tokens,
and an image ``fake://<name>/<n>`` is ``n`` placeholders whose pixel rows are filled with
``len(name)``), and the engine is a fake ``/inference/v1/generate`` that answers every prompt ``ok``.

uv run --isolated --extra dev --extra skyrl-train --extra skycap pytest tests/train/generators/test_skyrl_vlm_generator.py -v
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import aiohttp
import pytest
import pytest_asyncio
import torch
from aiohttp import web
from aiohttp.test_utils import TestServer
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers.skycap_engine import SkyRLEngine
from skyrl.train.config import (
    ChatTemplateConfig,
    GeneratorConfig,
    SamplingParams,
    SkyRLGymConfig,
)
from skyrl.train.generators.base import GeneratorInput
from skyrl.train.generators.skyrl_vlm_generator import SkyRLVLMGymGenerator
from skyrl_gym.envs import deregister, register
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.registration import registry

pytest.importorskip("skycap")

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
_spec = importlib.util.spec_from_file_location(
    "skycap_fake_renderer", Path(__file__).parents[3] / "skycap" / "tests" / "fake_renderer.py"
)
fake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake)

ANSWER = "ok"


def _image_message(name: str, placeholders: int, text: str) -> Dict[str, Any]:
    return {"role": "user", "content": [fake.image(name, placeholders), {"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# Test environments
# ---------------------------------------------------------------------------


class CPUVLMTestEnv(BaseTextEnv):
    """3-turn env with text observations; each turn's reward is its turn number."""

    def __init__(self, env_config: Any, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 3
        self.actions: List[str] = []

    def init(self, prompt):
        return prompt, {}

    def step(self, action: str):
        self.turns += 1
        self.actions.append(action)
        done = self.turns >= self.max_turns
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": f"turn {self.turns}"}] if not done else [],
            reward=float(self.turns),
            done=done,
            metadata={},
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {"actions": self.actions}


class CPUVLMImageObsEnv(BaseTextEnv):
    """3-turn env that returns a new image observation after every step, including the last."""

    def __init__(self, env_config: Any, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 3

    def init(self, prompt):
        return [*prompt, _image_message("prompt", 3, "look")], {}

    def step(self, action: str):
        self.turns += 1
        done = self.turns >= self.max_turns
        return BaseTextEnvStepOutput(
            observations=[_image_message(f"obs{self.turns}", 2, f"step {self.turns}")],
            reward=1.0 if done else 0.0,
            done=done,
            metadata={},
        )


for _env_id, _cls in (("cpu_vlm_test_env", "CPUVLMTestEnv"), ("cpu_vlm_image_obs_env", "CPUVLMImageObsEnv")):
    if _env_id not in registry:
        register(id=_env_id, entry_point=f"tests.train.generators.test_skyrl_vlm_generator:{_cls}")


@pytest.fixture(autouse=True, scope="module")
def deregister_test_env():
    yield
    deregister("cpu_vlm_test_env")
    deregister("cpu_vlm_image_obs_env")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------------------------
# A fake engine and a generator in front of it
# ---------------------------------------------------------------------------


class FakeEngine:
    """vLLM's ``/inference/v1/generate`` behind SkyRL's router: answers ``ok`` then the stop token, logprob -0.5 per token."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.sessions: List[str] = []
        self.released: List[str] = []

    def app(self) -> web.Application:
        app = web.Application(client_max_size=1024**3)
        app.router.add_post("/inference/v1/generate", self.generate)
        app.router.add_post("/finish_session", self.finish_session)
        return app

    async def generate(self, request: web.Request) -> web.Response:
        self.requests.append(await request.json())
        self.sessions.append(request.headers["X-Session-ID"])
        completion = [*fake.encode(ANSWER), fake.END]
        content = [{"logprob": -0.5} for _ in completion]
        choice = {"index": 0, "token_ids": completion, "logprobs": {"content": content}, "finish_reason": "stop"}
        return web.json_response({"choices": [choice]})

    async def finish_session(self, request: web.Request) -> web.Response:
        self.released.append(request.query["session_id"])
        return web.json_response({})


class FakeRendererGenerator(SkyRLVLMGymGenerator):
    def _capture_options(self) -> Dict[str, Any]:
        self.renderer = fake.FakeRenderer()
        return dict(renderer=self.renderer, engine=SkyRLEngine(packed_side_channels=False))


@pytest_asyncio.fixture
async def engine():
    engine = FakeEngine()
    server = TestServer(engine.app())
    await server.start_server()
    engine.url = str(server.make_url("")).rstrip("/")
    yield engine
    await server.close()


def _generator_cfg(**overrides) -> GeneratorConfig:
    kwargs = dict(
        sampling_params=SamplingParams(max_generate_length=200, logprobs=None),
        max_input_length=4096,
        batched=False,
        max_turns=3,
        zero_reward_on_non_stop=False,
        apply_overlong_filtering=False,
        use_conversation_multi_turn=True,
        chat_template=ChatTemplateConfig(source="name", name_or_path=None),
        step_wise_trajectories=False,
        vision_language_generator=True,
    )
    kwargs.update(overrides)
    return GeneratorConfig(**kwargs)


def _build_generator(tokenizer, engine_url: str = "http://unused", **cfg_overrides) -> FakeRendererGenerator:
    client = MagicMock()
    client.model_name = MODEL_NAME
    client.get_endpoint_url.return_value = engine_url
    return FakeRendererGenerator(
        generator_cfg=_generator_cfg(**cfg_overrides),
        skyrl_gym_cfg=SkyRLGymConfig(max_env_workers=0),
        inference_engine_client=client,
        tokenizer=tokenizer,
    )


def _input_batch(env_class: str = "cpu_vlm_test_env", sampling_params=None) -> GeneratorInput:
    batch: GeneratorInput = {
        "prompts": [[{"role": "user", "content": "a"}]],
        "env_extras": [{"answer": "4"}],
        "env_classes": [env_class],
    }
    if sampling_params is not None:
        batch["sampling_params"] = sampling_params
    return batch


def _runs(mask: List[int]) -> List[List[int]]:
    """Index runs of 1s."""
    runs: List[List[int]] = []
    for i, bit in enumerate(mask):
        if bit and (i == 0 or not mask[i - 1]):
            runs.append([])
        if bit:
            runs[-1].append(i)
    return runs


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"step_wise_trajectories": True}, "step-wise"),
        ({"use_conversation_multi_turn": False}, "use_conversation_multi_turn"),
        ({"chat_template": ChatTemplateConfig(source="name", name_or_path="qwen3_without_thinking")}, "custom chat"),
        ({"batched": True}, "batched"),
    ],
)
def test_vlm_validate_cfg_refusals(tokenizer, overrides, match):
    with pytest.raises(ValueError, match=match):
        _build_generator(tokenizer, **overrides)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vlm_rollout_is_the_captured_path_token_for_token(tokenizer, engine):
    generator = _build_generator(tokenizer, engine.url)
    output = await generator.generate(_input_batch())
    prompt, response, mask = output["prompt_token_ids"][0], output["response_ids"][0], output["loss_masks"][0]
    completion = [*fake.encode(ANSWER), fake.END]

    # Every turn extended the previous turn's exact prompt and completion.
    assert len(engine.requests) == 3
    for previous, current in zip(engine.requests, engine.requests[1:]):
        assert (
            current["token_ids"][: len(previous["token_ids"]) + len(completion)] == previous["token_ids"] + completion
        )
    # The response starts at the first sampled token and ends with the last reply.
    assert prompt == engine.requests[0]["token_ids"]
    assert prompt + response == engine.requests[-1]["token_ids"] + completion
    assert [[response[i] for i in run] for run in _runs(mask)] == [completion] * 3
    # The env saw the reply's text, without the stop token.
    assert output["env_metrics"][0]["actions"] == [ANSWER] * 3
    # Each turn's reward is on its reply's last token.
    rewards = output["rewards"][0]
    assert [rewards[run[-1]] for run in _runs(mask)] == [1.0, 2.0, 3.0]
    assert sum(rewards) == 6.0
    assert output["stop_reasons"] == ["stop"]
    assert output["rollout_logprobs"] is None
    # One engine session per trajectory, released when it finished.
    assert len(set(engine.sessions)) == 1 and engine.released == [engine.sessions[0]]


@pytest.mark.asyncio
async def test_vlm_rollout_logprobs_are_the_engines_on_sampled_tokens(tokenizer, engine):
    generator = _build_generator(
        tokenizer, engine.url, sampling_params=SamplingParams(max_generate_length=200, logprobs=0)
    )
    output = await generator.generate(_input_batch())
    logprobs, mask = output["rollout_logprobs"][0], output["loss_masks"][0]

    assert len(logprobs) == len(mask)
    assert all(lp == -0.5 for lp, bit in zip(logprobs, mask) if bit)
    assert all(lp == 0.0 for lp, bit in zip(logprobs, mask) if not bit)


@pytest.mark.asyncio
async def test_vlm_images_reach_the_engine_and_the_trainer(tokenizer, engine):
    generator = _build_generator(tokenizer, engine.url)
    output = await generator.generate(_input_batch("cpu_vlm_image_obs_env"))

    # Every call carries all the images of its prompt; the final observation's is never sent.
    hashes = [request.get("features", {}).get("mm_hashes", {}).get("image") for request in engine.requests]
    assert hashes == [["prompt"], ["prompt", "obs1"], ["prompt", "obs1", "obs2"]]
    last = engine.requests[-1]
    for placeholder in last["features"]["mm_placeholders"]["image"]:
        span = last["token_ids"][placeholder["offset"] : placeholder["offset"] + placeholder["length"]]
        assert span == [fake.IMG] * placeholder["length"]
    # The trainer gets those three images' arrays, in order, matching the placeholders in the sequence.
    pixel_values, image_grid_thw = output["pixel_values"][0], output["image_grid_thw"][0]
    assert isinstance(pixel_values, torch.Tensor)
    assert image_grid_thw.tolist() == [[1, 1, 3], [1, 1, 2], [1, 1, 2]]
    assert pixel_values[:, 0].tolist() == [6.0] * 3 + [4.0] * 2 + [4.0] * 2
    sequence = output["prompt_token_ids"][0] + output["response_ids"][0]
    assert sequence.count(fake.IMG) == len(pixel_values)
    assert all(bit == 0 for token, bit in zip(output["response_ids"][0], output["loss_masks"][0]) if token == fake.IMG)


@pytest.mark.asyncio
async def test_vlm_sampling_params_reach_the_engine(tokenizer, engine):
    generator = _build_generator(tokenizer, engine.url)
    sampling_params = {
        "max_tokens": 77,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.05,
        "min_tokens": 1,
        "logprobs": None,
        "skip_special_tokens": True,
        "include_stop_str_in_output": True,
    }
    await generator.generate(_input_batch(sampling_params=sampling_params))
    sent = engine.requests[0]["sampling_params"]

    assert {key: sent[key] for key in ("max_tokens", "temperature", "top_p", "top_k", "min_p", "min_tokens")} == {
        "max_tokens": 77,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.05,
        "min_tokens": 1,
    }


@pytest.mark.asyncio
async def test_vlm_config_sampling_params_apply_when_the_batch_has_none(tokenizer, engine):
    generator = _build_generator(
        tokenizer, engine.url, sampling_params=SamplingParams(max_generate_length=55, temperature=0.3, logprobs=None)
    )
    await generator.generate(_input_batch())
    sent = engine.requests[0]["sampling_params"]

    assert sent["max_tokens"] == 55 and sent["temperature"] == 0.3


@pytest.mark.asyncio
async def test_vlm_unsupported_sampling_params_are_refused(tokenizer, engine):
    generator = _build_generator(tokenizer, engine.url)
    with pytest.raises(ValueError, match="best_of"):
        await generator.generate(_input_batch(sampling_params={"max_tokens": 10, "best_of": 2}))


@pytest.mark.asyncio
async def test_vlm_a_prompt_over_max_input_length_ends_the_trajectory(tokenizer, engine):
    probe = _build_generator(tokenizer, engine.url)
    await probe.generate(_input_batch())
    first_prompt = len(engine.requests[0]["token_ids"])
    engine.requests.clear()

    generator = _build_generator(tokenizer, engine.url, max_input_length=first_prompt)
    output = await generator.generate(_input_batch())

    assert len(engine.requests) == 1
    assert output["stop_reasons"] == ["length"]
    assert len(_runs(output["loss_masks"][0])) == 1
    assert output["prompt_token_ids"][0] + output["response_ids"][0] == engine.requests[0]["token_ids"] + [
        *fake.encode(ANSWER),
        fake.END,
    ]


@pytest.mark.asyncio
async def test_vlm_a_first_prompt_over_max_input_length_is_an_error(tokenizer, engine):
    generator = _build_generator(tokenizer, engine.url, max_input_length=3)
    with pytest.raises(ValueError, match="max_input_length"):
        await generator.generate(_input_batch())
    assert engine.requests == []


@pytest.mark.asyncio
async def test_vlm_chat_template_kwargs_and_processor_kwargs_reach_the_renderer(tokenizer):
    generator = _build_generator(tokenizer, chat_template_kwargs={"enable_thinking": False})
    generator.generator_cfg.inference_engine.engine_init_kwargs = {"mm_processor_kwargs": {"max_pixels": 1024}}
    generator.generator_cfg.vision_language_renderer = "qwen3-vl"
    options = SkyRLVLMGymGenerator._capture_options(generator)

    assert options["tokenizer"] == tokenizer.name_or_path
    assert options["renderer_name"] == "qwen3-vl"
    assert options["chat_template_kwargs"] == {"enable_thinking": False}
    assert options["processor_kwargs"] == {"max_pixels": 1024}
    assert isinstance(options["engine"], SkyRLEngine)
    assert options["engine"].generate_path == "/inference/v1/generate"


def test_the_packed_skyrl_route_refuses_images_it_would_drop():
    from skycap.tokens.engine import EngineError

    common = dict(prompt_ids=[1, 2], sampling={}, model=None, cache_salt=None, sampling_mask=False)
    features = {"mm_hashes": {"image": ["a"]}}
    with pytest.raises(EngineError, match="drops multimodal features"):
        SkyRLEngine().request(features=features, **common)
    assert SkyRLEngine(packed_side_channels=False).request(features=features, **common)["features"] == features


@pytest.mark.asyncio
async def test_vlm_trajectories_are_not_kept_once_finished(tokenizer, engine):
    generator = _build_generator(tokenizer, engine.url)
    await generator.generate(_input_batch())
    async with aiohttp.ClientSession() as http:
        async with http.get(f"{generator.capture_url}/trajectories/{engine.sessions[0]}") as response:
            assert response.status == 404
    assert generator.capture.server.trajectories == {}
