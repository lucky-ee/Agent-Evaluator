import re

import pytest

from problemgenerator import generate_problem


def replay(prompt):
    # independent check: replay the steps from the text and compare to the answer
    v = int(re.search(r"Start with (-?\d+)", prompt).group(1))
    for kind, n in re.findall(r"Step \d+: (add|subtract|multiply by|divide by) (\d+)", prompt):
        n = int(n)
        if kind == "add": v += n
        elif kind == "subtract": v -= n
        elif kind == "multiply by": v *= n
        else:
            assert v % n == 0, "division must be exact"
            v //= n
    return v


def test_same_seed_same_task():
    assert generate_problem(4, 2, seed=0) == generate_problem(4, 2, seed=0)


def test_different_seeds_differ():
    assert generate_problem(4, 2, seed=1).prompt != generate_problem(4, 2, seed=0).prompt


@pytest.mark.parametrize("h", [1, 4, 8, 20])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_replay_and_checkpoints(h, d):
    for s in range(200):
        task = generate_problem(h, d, s)
        assert replay(task.prompt) == task.expected_answer, (h, d, s)
        assert " to get " not in task.prompt, "prompt must not print running totals"
        assert len(task.checkpoints) == h, (h, d, s)
        assert task.checkpoints[-1] == task.expected_answer, (h, d, s)


def test_task_metadata():
    task = generate_problem(8, 3, seed=5)
    assert task.id == "h8-d3-s5"
    assert task.knobs == {"horizon": 8, "difficulty": 3, "seed": 5}
    assert task.allowed_tools == set()
    assert isinstance(task.expected_answer, int)
    assert task.prompt.endswith("End your response with a line of the form: ANSWER: <integer>")
