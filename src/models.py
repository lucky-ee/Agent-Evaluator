from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from grader import Task, ToolCall


@dataclass
class ModelResponse:
    content: str
    tool_call: ToolCall | None = None
    done: bool = False


class Model(Protocol):
    def step(self, messages: list[dict]) -> ModelResponse:
        """messages: [{"role": "system"|"user"|"assistant"|"tool", "content": str}, ...]"""
        ...


_FAKE_TOOLS = ["wolfram_alpha", "python_exec", "web_search", "sympy_solve"]


class FakeModel:
    def __init__(
        self,
        task: Task,
        seed: int,
        p_step_error: float = 0.0,
        p_hallucinate_tool: float = 0.0,
        p_stall: float = 0.0,
        p_malformed: float = 0.0,
    ):
        rng = random.Random(seed)
        h = len(task.checkpoints)

        # Draw the same number of values whatever the probabilities are, so e.g.
        # turning on tool hallucination doesn't change where the arithmetic slips are.
        self.values: list[int] = []
        self.bad_tools: list[str | None] = []
        offset = 0
        for cp in task.checkpoints:
            slip = rng.random() < p_step_error
            magnitude = rng.randint(1, 10)
            sign = rng.choice([-1, 1])
            hallucinate = rng.random() < p_hallucinate_tool
            tool = rng.choice(_FAKE_TOOLS)
            if slip:
                if offset != 0:
                    sign = 1 if offset > 0 else -1
                offset += sign * magnitude
            self.values.append(cp + offset)
            self.bad_tools.append(tool if hallucinate else None)

        stall = rng.random() < p_stall
        stall_at = rng.randint(0, h)
        self.stall_at: int | None = stall_at if stall else None
        self.malformed = rng.random() < p_malformed

    def step(self, messages: list[dict]) -> ModelResponse:
        i = sum(1 for m in messages if m["role"] == "assistant")

        if self.stall_at is not None and i >= self.stall_at:
            return ModelResponse("still working...")

        if i < len(self.values):
            content = f"The running value is now {self.values[i]}."
            tool = self.bad_tools[i]
            if tool is not None:
                return ModelResponse(content, ToolCall(tool, {"query": "check my arithmetic"}))
            return ModelResponse(content)

        final = self.values[-1] if self.values else 0
        if self.malformed:
            return ModelResponse(f"The final value is {final}.", done=True)
        return ModelResponse(f"ANSWER: {final}", done=True)
