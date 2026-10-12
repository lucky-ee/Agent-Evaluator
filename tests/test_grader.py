import pytest

from grader import (
    Failure, Outcome, Step, Task, ToolCall, Trajectory,
    answer_matches, grade, parse_answer, partial_credit,
)

TASK = Task(
    id="t1",
    prompt="Compute the chain...",
    expected_answer=42,
    checkpoints=[3, 7, 21, 42],
    allowed_tools={"calculator"},
)


def traj(steps, final="Final answer: 42", termination="finished"):
    return Trajectory("t1", [Step(s) if isinstance(s, str) else s for s in steps], final, termination)


GOOD_STEPS = ["start with 3", "add 4 to get 7", "times 3 is 21", "double: 42"]


# outcome classification tests

def test_correct():
    r = grade(TASK, traj(GOOD_STEPS))
    assert r.outcome is Outcome.CORRECT and r.score == 1.0 and r.failure is None


def test_step_limit_gets_partial_credit():
    r = grade(TASK, traj(GOOD_STEPS[:3], final="", termination="step_limit"))
    assert r.failure is Failure.STEP_LIMIT
    assert r.outcome is Outcome.PARTIAL
    assert r.score == pytest.approx(0.75)


def test_step_limit_with_all_checkpoints_is_not_full_credit():
    r = grade(TASK, traj(GOOD_STEPS, final="", termination="step_limit"))
    assert r.failure is Failure.STEP_LIMIT and r.score < 1.0


def test_context_overflow():
    r = grade(TASK, traj(["start with 3"], final="", termination="context_overflow"))
    assert r.failure is Failure.CONTEXT_OVERFLOW


def test_runtime_error():
    r = grade(TASK, traj([], final="", termination="error"))
    assert r.failure is Failure.RUNTIME_ERROR and r.outcome is Outcome.WRONG


def test_hallucinated_tool():
    steps = GOOD_STEPS[:2] + [Step("look it up", ToolCall("wolfram_alpha", {}, "21"))]
    r = grade(TASK, traj(steps))
    assert r.failure is Failure.HALLUCINATED_TOOL
    assert r.details["bad_tools"] == ["wolfram_alpha"]


def test_allowed_tool_is_fine():
    steps = [Step("3*7", ToolCall("calculator", {"expr": "3*7"}, 21))]
    r = grade(TASK, traj(["start with 3", "add 4 to get 7"] + steps + ["double: 42"]))
    assert r.outcome is Outcome.CORRECT


def test_malformed():
    r = grade(TASK, traj(GOOD_STEPS, final="I think it's 42"))
    assert r.failure is Failure.MALFORMED


def test_wrong_answer_with_partial_credit():
    r = grade(TASK, traj(GOOD_STEPS[:2] + ["times 3 is 20"], final="Final answer: 40"))
    assert r.failure is Failure.WRONG_ANSWER
    assert r.score == pytest.approx(0.5)
    assert r.details["got"] == "40"


# tricky answer parsing cases

def test_odd_formatting_still_correct():
    r = grade(TASK, traj(GOOD_STEPS, final="So... FINAL ANSWER:   42.0."))
    assert r.outcome is Outcome.CORRECT


def test_boxed_answer():
    r = grade(TASK, traj(GOOD_STEPS, final=r"Therefore \boxed{42}"))
    assert r.outcome is Outcome.CORRECT


def test_last_answer_wins():
    r = grade(TASK, traj(GOOD_STEPS, final="Final answer: 42\nWait, no. Final answer: 41"))
    assert r.failure is Failure.WRONG_ANSWER


def test_right_number_buried_in_wrong_answer():
    r = grade(TASK, traj(GOOD_STEPS, final="Final answer: 420 (not 42)"))
    assert r.failure is Failure.WRONG_ANSWER


def test_prefix_credit_ignores_lucky_later_steps():
    steps = ["start with 3", "add 4 to get 8", "times 3 is 21", "double: 42"]
    score, hit = partial_credit(traj(steps), TASK.checkpoints, "prefix")
    assert hit == 1
    score_f, hit_f = partial_credit(traj(steps), TASK.checkpoints, "fraction")
    assert hit_f == 3


def test_wrong_task_id_raises():
    with pytest.raises(ValueError):
        grade(TASK, Trajectory("other", [], "", "finished"))


# answer_matches() is used in partial credit and tool hallucination checks, so we test it here too.

@pytest.mark.parametrize("got,expected", [
    ("1/2", 0.5), ("0.5", "1/2"), ("1,000", 1000), ("1e3", 1000),
    ("$42$", 42), (" Yes ", "yes"), ("0.30000000000000004", 0.3),
])
def test_matches(got, expected):
    assert answer_matches(got, expected)


@pytest.mark.parametrize("got,expected", [
    ("0.5001", 0.5), ("41", 42), ("no", "yes"), ("1/3", 0.33),
])
def test_does_not_match(got, expected):
    assert not answer_matches(got, expected)


def test_parse_answer_none():
    assert parse_answer("") is None
    assert parse_answer("Final answer: ") is None