import pytest

from grader import Failure, Outcome, Step, Trajectory, grade
from problemgenerator import generate_problem

COMBOS = [(1, 1, 0), (4, 2, 7), (8, 1, 3), (8, 3, 11), (20, 2, 42), (20, 3, 0), (20, 3, 99)]


def steps_for(checkpoints):
    return [Step(f"The value is now {cp}.") for cp in checkpoints]


@pytest.fixture(params=COMBOS, ids=lambda c: f"h{c[0]}-d{c[1]}-s{c[2]}")
def task(request):
    return generate_problem(*request.param)


def test_perfect_run(task):
    traj = Trajectory(task.id, steps_for(task.checkpoints), f"ANSWER: {task.expected_answer}", "finished")
    r = grade(task, traj)
    assert r.outcome is Outcome.CORRECT and r.score == 1.0 and r.failure is None


def test_halfway_step_limit(task):
    half = len(task.checkpoints) // 2
    traj = Trajectory(task.id, steps_for(task.checkpoints[:half]), "", "step_limit")
    r = grade(task, traj)
    assert r.failure is Failure.STEP_LIMIT
    assert r.score == pytest.approx(half / len(task.checkpoints))


def test_wrong_final_answer(task):
    traj = Trajectory(task.id, steps_for(task.checkpoints), f"ANSWER: {task.expected_answer + 1}", "finished")
    r = grade(task, traj)
    assert r.failure is Failure.WRONG_ANSWER


def test_missing_answer_line(task):
    traj = Trajectory(task.id, steps_for(task.checkpoints), str(task.expected_answer), "finished")
    r = grade(task, traj)
    assert r.failure is Failure.MALFORMED
