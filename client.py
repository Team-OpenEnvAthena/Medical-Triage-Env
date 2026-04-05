# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Medical Triage Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import TriageAction, TriageObservation, TriageState


class MedicalTriageEnv(EnvClient[TriageAction, TriageObservation, TriageState]):
    """
    Client for the Medical Triage & Discharge Planning Environment.

    Connects via WebSocket (/ws) for persistent, stateful sessions.

    Usage (async):
        async with MedicalTriageEnv(base_url="https://...hf.space") as env:
            result = await env.reset(task="easy")
            result = await env.step(TriageAction(task_type="easy", urgency_assignment=1))
            print(result.reward)          # float in [0.0, 1.0]
            print(result.observation.partial_score)  # shaped reward signal
            state = await env.state()
            print(state.current_task)     # "easy" — for TRL/GRPO reward routing

    Usage (sync):
        with MedicalTriageEnv(base_url="https://...hf.space").sync() as env:
            result = env.reset(task="hard")
            result = env.step(TriageAction(
                task_type="hard",
                diagnosis="pneumonia",
                disposition="admit",
                prescribed_medications=["antibiotics"],
                follow_up_days=1
            ))

    Usage from Docker image:
        client = await MedicalTriageEnv.from_docker_image("medical-triage-env:latest")
        async with client:
            result = await client.reset()
    """

    def _step_payload(self, action: TriageAction) -> Dict:
        """
        Convert TriageAction to WebSocket step payload.
        Returns flat action fields — the EnvClient WebSocket protocol
        does not use an {"action": ...} wrapper.
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[TriageObservation]:
        """Parse server WebSocket response into StepResult[TriageObservation]."""
        obs_data = payload.get("observation", {})

        # Reward may be at top-level (standard OpenEnv) OR inside the observation
        # (when environment.step() returns the Observation directly and the
        # framework serializes reward as part of the obs dict).
        # Check both locations so we never silently return 0.0.
        reward = (
            payload.get("reward")
            if payload.get("reward") is not None
            else obs_data.get("reward", 0.0)
        )
        done = payload.get("done", obs_data.get("done", False))

        observation = TriageObservation(
            current_patient=obs_data.get("current_patient"),
            available_investigations=obs_data.get("available_investigations", []),
            investigation_results=obs_data.get("investigation_results"),
            task_instruction=obs_data.get("task_instruction", ""),
            partial_score=obs_data.get("partial_score", 0.0),
            safety_flags=obs_data.get("safety_flags", []),
            done=done,
            reward=reward,
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict) -> TriageState:
        """
        Parse server state response into TriageState.

        TriageState extends base State (episode_id, step_count) with
        current_task — used by TRL/GRPO to route task-specific reward functions.
        """
        return TriageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_task=payload.get("current_task", "easy"),
        )
