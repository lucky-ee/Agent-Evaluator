import ast
import dataclasses
import operator
import re

import pytest

from grader import Failure, Outcome, ToolCall, grade
from models import FakeModel, ModelResponse
from problemgenerator import generate_problem
from runner import run

COMBOS = [(1, 1, 0), (4, 2, 7), (8, 1, 3), (8, 3, 11), (20, 2, 42), (20, 3, 0), (20, 3, 99)]


@pytest.fixture(params=COMBOS, ids=lambda c: f"h{c[0]}-d{c[1]}-s{c[2]}")
def task(request):
    return generate_problem(*request.param)


def run_and_grade(task, model, **kwargs):
    traj = run(task, model, **kwargs)
    return traj, grade(task, traj)


# one test per outcome

def test_perfect_model_is_correct(task):
    traj, r = run_and_grade(task, FakeModel(task, seed=0))
    assert traj.termination == "finished"
    assert r.outcome is Outcome.CORRECT and r.score == 1.0 and r.failure is None


def test_step_errors_give_wrong_answer(task):
    traj, r = run_and_grade(task, FakeModel(task, seed=0, p_step_error=1.0))
    assert traj.termination == "finished"
    assert r.failure is Failure.WRONG_ANSWER
    assert r.score < 1.0


def test_late_step_error_gets_partial_credit():
    # Find a run whose first slip comes after step 1, so some checkpoints are hit.
    task = generate_problem(8, 2, seed=1)
    for seed in range(100):
        model = FakeModel(task, seed=seed, p_step_error=0.2)
        if model.values[0] == task.checkpoints[0] and model.values[-1] != task.expected_answer:
            break
    else:
        pytest.fail("no seed with a late slip")
    _, r = run_and_grade(task, model)
    assert r.failure is Failure.WRONG_ANSWER
    assert r.outcome is Outcome.PARTIAL and 0 < r.score < 1.0


def test_stall_hits_step_limit(task):
    traj, r = run_and_grade(task, FakeModel(task, seed=0, p_stall=1.0), max_steps=30)
    assert traj.termination == "step_limit" and traj.final_output == ""
    assert len(traj.steps) == 30
    assert r.failure is Failure.STEP_LIMIT


def test_tiny_context_overflows(task):
    traj, r = run_and_grade(task, FakeModel(task, seed=0), max_context_chars=len(task.prompt) + 10)
    assert traj.termination == "context_overflow" and traj.final_output == ""
    assert r.failure is Failure.CONTEXT_OVERFLOW


def test_hallucinated_tool(task):
    traj, r = run_and_grade(task, FakeModel(task, seed=0, p_hallucinate_tool=1.0))
    assert traj.termination == "finished"
    assert r.failure is Failure.HALLUCINATED_TOOL
    call = traj.steps[0].tool_call
    assert call is not None and call.result == f"error: unknown tool {call.name}"


def test_malformed_answer(task):
    traj, r = run_and_grade(task, FakeModel(task, seed=0, p_malformed=1.0))
    assert traj.termination == "finished"
    assert r.failure is Failure.MALFORMED


class CrashingModel:
    def __init__(self, inner, crash_after):
        self.inner, self.crash_after, self.calls = inner, crash_after, 0

    def step(self, messages):
        self.calls += 1
        if self.calls > self.crash_after:
            raise RuntimeError("backend exploded")
        return self.inner.step(messages)


def test_model_exception_is_runtime_error():
    task = generate_problem(8, 2, seed=5)
    traj, r = run_and_grade(task, CrashingModel(FakeModel(task, seed=0), crash_after=3))
    assert traj.termination == "error" and traj.final_output == ""
    assert len(traj.steps) == 4
    assert "backend exploded" in traj.steps[-1].content
    assert r.failure is Failure.RUNTIME_ERROR
    assert r.details["checkpoints_hit"] == 3


# determinism

NOISY = dict(p_step_error=0.2, p_hallucinate_tool=0.2, p_stall=0.2, p_malformed=0.2)


def test_same_seed_same_trajectory():
    task = generate_problem(20, 2, seed=3)
    assert run(task, FakeModel(task, seed=7, **NOISY)) == run(task, FakeModel(task, seed=7, **NOISY))


def test_different_seeds_can_differ():
    task = generate_problem(20, 2, seed=3)
    base = run(task, FakeModel(task, seed=0, **NOISY))
    assert any(run(task, FakeModel(task, seed=s, **NOISY)) != base for s in range(1, 10))


# allowed tools

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}


def calculator(expression):
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError(f"unsupported expression: {expression!r}")
    return ev(ast.parse(expression, mode="eval").body)


class CalculatorModel:
    """Wraps FakeModel and checks each step's value with the calculator tool."""

    def __init__(self, inner):
        self.inner = inner

    def step(self, messages):
        r = self.inner.step(messages)
        if r.tool_call is None and not r.done:
            value = int(re.findall(r"-?\d+", r.content)[-1])
            return ModelResponse(r.content, ToolCall("calculator", {"expression": f"{value} * 1"}))
        return r


def test_allowed_tool_runs_and_stays_correct():
    task = generate_problem(8, 2, seed=4)
    task = dataclasses.replace(task, allowed_tools={"calculator"})
    traj, r = run_and_grade(task, CalculatorModel(FakeModel(task, seed=0)), tools={"calculator": calculator})
    results = [s.tool_call.result for s in traj.steps if s.tool_call is not None]
    assert results == task.checkpoints
    assert r.outcome is Outcome.CORRECT and r.score == 1.0


# statistical sanity check: accuracy should follow (1 - p)^h

@pytest.mark.parametrize("p,h", [(0.1, 5), (0.05, 10), (0.02, 20), (0.2, 4)])
def test_accuracy_follows_compounding_error(p, h):
    n = 500
    correct = 0
    for s in range(n):
        task = generate_problem(h, 2, seed=s)
        _, r = run_and_grade(task, FakeModel(task, seed=s, p_step_error=p))
        correct += r.outcome is Outcome.CORRECT
    assert correct / n == pytest.approx((1 - p) ** h, abs=0.07)
