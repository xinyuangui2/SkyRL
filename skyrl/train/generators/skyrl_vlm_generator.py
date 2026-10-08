"""
SkyRLVLMGymGenerator: VLM (vision-language model) multi-turn RL generator.
"""

import copy
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import torch
from loguru import logger

import skyrl_gym
from skyrl.backends.skyrl_train.inference_servers.base import ConversationType
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import (
    TRAINING_PHASE_TRAIN,
    GeneratorInput,
    GeneratorOutput,
    TrainingPhase,
    TrajectoryID,
)
from skyrl.train.generators.skyrl_gym_generator import (
    SkyRLGymGenerator,
    TrajectoryOutput,
)

#: Engine sampling params a chat request to skycap carries. The rest are fixed by token-in/token-out
#: capture (``logprobs``: the sampled token's are always captured) or don't apply to token ids
#: (``skip_special_tokens``, ``include_stop_str_in_output``).
CHAT_SAMPLING_KEYS = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "min_tokens",
    "seed",
    "stop",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
)
IGNORED_SAMPLING_KEYS = frozenset({"logprobs", "skip_special_tokens", "include_stop_str_in_output"})

#: skycap's error code for a prompt over the request's ``max_prompt_tokens``.
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"


class SkyRLVLMGymGenerator(SkyRLGymGenerator):
    """VLM generator that handles multi-modal (text + image) observations.

    Rollouts are captured by skycap in token mode. The generator runs a skycap server in this process,
    in front of the inference engine, and speaks OpenAI chat to it: each turn sends the conversation so
    far, images included, and skycap renders it, extends the previous turn's exact prompt and completion
    tokens with the new messages, and sends the engine those token ids and every image in the prompt.
    When the trajectory ends, skycap returns its tokens, loss mask, rollout logprobs and images, which
    become the trajectory's output.
    """

    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client: RemoteInferenceClient,
        tokenizer,
        policy_model_name: Optional[str] = None,
    ):
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, policy_model_name)
        self.model_name = policy_model_name or getattr(inference_engine_client, "model_name", None)
        self.capture_url = self._start_capture()
        #: Running totals since the generator started, reported as ``skycap/*`` rollout metrics. A call is
        #: unbridged when skycap rendered its prompt from the messages instead of extending the previous
        #: call's exact tokens; the agent loop's history is append-only, so that stays 0.
        self.capture_stats = {"trajectories": 0, "calls": 0, "unbridged_calls": 0}
        logger.info(f"Initialized SkyRLVLMGymGenerator, capturing with skycap at {self.capture_url}")

    def _start_capture(self) -> str:
        """Start the skycap server this generator's trajectories are captured by. Returns its URL."""
        from skycap import CaptureService

        self.capture = CaptureService(
            self.inference_engine_client.get_endpoint_url(),
            mode="tokens",
            # Replies are the completion's text, as the engine decodes it: an env parses its own actions
            # (tool calls included) out of the text, and reasoning stays inline.
            use_raw_content=True,
            # Nothing is recorded: a trajectory is dropped once finished, and its samples are what it returned.
            keep_unrecorded=False,
            host="127.0.0.1",
            **self._capture_options(),
        )
        return self.capture.start()

    def _capture_options(self) -> Dict[str, Any]:
        """How skycap renders and reaches the engine: the model's ``renderers`` renderer and SkyRL's wire."""
        from skyrl.backends.skyrl_train.inference_servers.skycap_engine import (
            SkyRLEngine,
        )

        engine_kwargs = self.generator_cfg.inference_engine.engine_init_kwargs or {}
        return dict(
            tokenizer=self.tokenizer.name_or_path,
            renderer_name=self.generator_cfg.vision_language_renderer,
            chat_template_kwargs=self.generator_cfg.chat_template_kwargs or None,
            processor_kwargs=engine_kwargs.get("mm_processor_kwargs"),
            # /skyrl/v1/generate drops images; VLM runs need no packed side channels (no R3, no sample support).
            engine=SkyRLEngine(packed_side_channels=False),
        )

    def _validate_cfg(self, generator_cfg: GeneratorConfig):
        if generator_cfg.batched:
            raise ValueError("SkyRLVLMGymGenerator does not support batched generation. Set `batched=False`.")
        if generator_cfg.step_wise_trajectories:
            raise ValueError("SkyRLVLMGymGenerator does not support step-wise trajectories.")
        if not generator_cfg.use_conversation_multi_turn:
            raise ValueError(
                "SkyRLVLMGymGenerator requires `use_conversation_multi_turn=True` "
                "because multi-modal observations must be in separate user messages."
            )
        if self.custom_chat_template is not None:
            raise ValueError(
                "SkyRLVLMGymGenerator does not support a custom chat template, got "
                f"{generator_cfg.chat_template}. skycap renders with the model's renderer."
            )
        super()._validate_cfg(generator_cfg)

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        trajectory_id: Optional[TrajectoryID] = None,
        cache_salt: Optional[str] = None,
        training_phase: TrainingPhase = TRAINING_PHASE_TRAIN,
    ) -> TrajectoryOutput:
        """Multi-turn VLM generation loop for a single trajectory.

        Each turn sends the whole conversation to the trajectory's skycap URL, with the turn's prompt
        bounded by ``max_input_length``; a longer one ends the trajectory with ``stop_reason="length"``.
        The trajectory is finished with skycap's ``final`` path rule: the path to the last reply, every
        reply on it trained. Observations after the last reply are never sent, so the final observation
        and its images are not part of the output.
        """
        from skycap import CapturePool

        agent_loop_start_time = time.monotonic()
        time_splits = {"llm": 0.0, "env": 0.0}

        env_extras["max_turns"] = self.max_turns
        env_extras = self._setup_env_extras(env_class, env_extras, sampling_params, trajectory_id)
        env_config = getattr(self.skyrl_gym_cfg, env_class, dict())
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        # As in SkyRLGymGenerator, `sampling_params` is None when the engine's defaults (the config's) apply.
        current_sampling_params: dict = (
            sampling_params if sampling_params is not None else asdict(self.generator_cfg.sampling_params)
        )
        get_logprobs = current_sampling_params.get("logprobs", None) is not None
        request = self._chat_request(sampling_params, max_input_length, cache_salt)

        conversation = copy.deepcopy(prompt)
        conversation, _ = await self._run_in_executor_if_available(env.init, conversation)

        meta = {"env_class": env_class}
        if trajectory_id is not None:
            meta.update(instance_id=str(trajectory_id.instance_id), repetition_id=trajectory_id.repetition_id)
        # Rewards of the turns that generated, in order: the i-th goes on the last token of the i-th reply.
        turn_rewards: List[float] = []
        stop_reason = "stop"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as http:
            async with CapturePool([self.capture_url]) as pool:
                async with pool.trajectory(meta, paths="final") as trajectory:
                    url = f"{trajectory.base_url}/chat/completions"
                    done = False
                    while not done:
                        llm_call_start_time = time.monotonic()
                        reply = await self._chat(http, url, {**request, "messages": conversation})
                        time_splits["llm"] += time.monotonic() - llm_call_start_time
                        if reply is None:
                            stop_reason = "length"
                            break
                        choice = reply["choices"][0]
                        stop_reason = choice["finish_reason"]
                        message = choice["message"]

                        env_step_start_time = time.monotonic()
                        env_step_output = await self._run_in_executor_if_available(env.step, message["content"])
                        time_splits["env"] += time.monotonic() - env_step_start_time
                        turn_rewards.append(env_step_output["reward"])
                        done = env_step_output["done"]
                        conversation = [*conversation, _replayed(message), *env_step_output["observations"]]
                    finished = await trajectory.finish()
        self.capture_stats["trajectories"] += 1
        self.capture_stats["calls"] += len(turn_rewards)
        self.capture_stats["unbridged_calls"] += finished.unbridged_calls

        env_metrics = env.get_metrics()
        await self._run_in_executor_if_available(env.close)
        if not turn_rewards:
            raise ValueError(
                f"trajectory {trajectory_id}: the prompt is longer than max_input_length={max_input_length} tokens"
            )
        if finished.status != "finished" or len(finished.samples) != 1:
            raise RuntimeError(
                f"skycap could not capture trajectory {trajectory_id} exactly: status {finished.status!r}, "
                f"{len(finished.samples)} samples"
            )
        (sample,) = finished.samples
        prompt_ids, response_ids, loss_mask, rollout_logprobs, reply_ends = _split_sample(sample)
        if len(reply_ends) != len(turn_rewards):
            raise RuntimeError(
                f"trajectory {trajectory_id}: {len(turn_rewards)} turns generated, but the captured path has "
                f"{len(reply_ends)} replies"
            )
        reward_out = self._build_per_token_rewards(
            list(zip(turn_rewards, reply_ends)), response_ids, appended_eos_token=False
        )
        pixel_values, image_grid_thw = _vision_features(sample.media)

        agent_loop_output = TrajectoryOutput(
            response_ids=response_ids,
            reward=reward_out,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            rollout_logprobs=rollout_logprobs if get_logprobs else None,
            env_metrics=env_metrics,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        agent_loop_output = self._post_process_agent_loop_output(agent_loop_output, env_extras, trajectory_id)
        agent_loop_output.e2e_time = time.monotonic() - agent_loop_start_time
        agent_loop_output.time_splits = time_splits
        return agent_loop_output

    def _chat_request(
        self, sampling_params: Optional[Dict[str, Any]], max_input_length: int, cache_salt: Optional[str]
    ) -> Dict[str, Any]:
        """The chat request body every turn of a trajectory shares, all but its messages.

        ``sampling_params`` are the engine's (``get_sampling_params_for_backend``), or None for the
        config's.
        """
        if sampling_params is None:
            from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
                get_sampling_params_for_backend,
            )

            sampling_params = get_sampling_params_for_backend(
                self.generator_cfg.inference_engine.backend, self.generator_cfg.sampling_params
            )
        unsupported = set(sampling_params) - set(CHAT_SAMPLING_KEYS) - IGNORED_SAMPLING_KEYS
        unsupported = {key for key in unsupported if sampling_params[key] is not None}
        if unsupported:
            raise ValueError(f"SkyRLVLMGymGenerator cannot send sampling params {sorted(unsupported)} through skycap")
        if sampling_params.get("n", 1) != 1:
            raise ValueError("n > 1 is not supported. Use `config.generator.n_samples_per_prompt` instead.")
        request: Dict[str, Any] = {
            key: sampling_params[key] for key in CHAT_SAMPLING_KEYS if sampling_params.get(key) is not None
        }
        request["model"] = self.model_name
        request["max_prompt_tokens"] = max_input_length
        if cache_salt is not None:
            request["cache_salt"] = cache_salt
        return request

    @staticmethod
    async def _chat(http: aiohttp.ClientSession, url: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """One chat completion, or None when the prompt is over ``max_prompt_tokens``."""
        async with http.post(url, json=body) as response:
            payload = await response.json(content_type=None)
            if response.status == 200:
                return payload
        error = payload.get("error") if isinstance(payload, dict) else None
        if response.status == 400 and isinstance(error, dict) and error.get("code") == CONTEXT_LENGTH_EXCEEDED:
            return None
        raise RuntimeError(f"skycap chat completion failed: HTTP {response.status}: {payload}")

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        output = await super().generate(input_batch, disable_tqdm=disable_tqdm)
        output["rollout_metrics"].update({f"skycap/{key}": value for key, value in self.capture_stats.items()})
        return output

    async def generate_batched(self, *args, **kwargs) -> GeneratorOutput:
        raise NotImplementedError(
            "SkyRLVLMGymGenerator does not support batched generation. "
            "Use the default async agent_loop path instead."
        )


def _replayed(message: Dict[str, Any]) -> Dict[str, Any]:
    """The reply as the next turn sends it back: the fields skycap answered with, ``None``s dropped."""
    return {key: value for key, value in message.items() if value is not None}


def _split_sample(sample: Any) -> Tuple[List[int], List[int], List[int], List[float], List[int]]:
    """A captured path as ``(prompt_ids, response_ids, loss_mask, rollout_logprobs, reply_ends)``.

    The response starts at the first sampled token, as in ``SkyRLGymGenerator``: the first reply's
    generation prompt is part of the prompt. ``reply_ends`` are the response indices of each reply's
    last sampled token. Replies are runs of trained tokens, separated by the observations and template
    scaffold between them.
    """
    loss_mask = list(sample.loss_mask)
    if 1 not in loss_mask:
        raise RuntimeError("the captured path has no sampled tokens")
    start = loss_mask.index(1)
    response_mask = loss_mask[start:]
    reply_ends = [
        i for i, bit in enumerate(response_mask) if bit and (i + 1 == len(response_mask) or not response_mask[i + 1])
    ]
    return (
        list(sample.input_ids[:start]),
        list(sample.input_ids[start:]),
        response_mask,
        list(sample.logprobs[start:]),
        reply_ends,
    )


def _vision_features(media: List[Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """The path's images as ``(pixel_values, image_grid_thw)`` in placeholder order, or ``(None, None)``."""
    images = [item for item in media if item.modality == "image"]
    if len(images) != len(media):
        raise ValueError(f"SkyRLVLMGymGenerator supports images only, got {sorted({m.modality for m in media})}")
    if not images:
        return None, None
    pixel_values = torch.from_numpy(np.concatenate([np.asarray(item.data["pixel_values"]) for item in images]))
    image_grid_thw = torch.from_numpy(np.concatenate([np.asarray(item.data["image_grid_thw"]) for item in images]))
    return pixel_values, image_grid_thw
