from __future__ import annotations

from typing import Any, Callable

from grader import Step, Task, Trajectory, parse_answer
from models import Model


def _context_chars(messages: list[dict]) -> int:
    # TODO: count tokens with the backend's real tokenizer; characters are a stand-in.
    return sum(len(m["content"]) for m in messages)


def run(
    task: Task,
    model: Model,
    max_steps: int = 50,
    max_context_chars: int = 20_000,
    tools: dict[str, Callable[..., Any]] | None = None,
) -> Trajectory:
    tools = tools or {}
    messages: list[dict] = [{"role": "user", "content": task.prompt}]
    steps: list[Step] = []

    def end(termination: str, final_output: str = "") -> Trajectory:
        return Trajectory(task.id, steps, final_output, termination)

    while len(steps) < max_steps:
        if _context_chars(messages) > max_context_chars:
            return end("context_overflow")

        try:
            response = model.step(messages)
        except Exception as e:
            steps.append(Step(f"error: {type(e).__name__}: {e}"))
            return end("error")

        step = Step(response.content, response.tool_call)
        steps.append(step)
        messages.append({"role": "assistant", "content": response.content})

        call = response.tool_call
        if call is not None:
            # Log the call even when the tool doesn't exist: the grader finds
            # hallucinated tools from the ToolCall on the Step.
            if call.name in tools:
                try:
                    call.result = tools[call.name](**call.args)
                except Exception as e:
                    call.result = f"error: {type(e).__name__}: {e}"
            else:
                call.result = f"error: unknown tool {call.name}"
            messages.append({"role": "tool", "content": str(call.result)})
            continue

        if response.done or parse_answer(response.content) is not None:
            return end("finished", response.content)

        messages.append({"role": "user", "content": "Continue."})

    return end("step_limit")
