import random

from grader import Task


def generate_problem(horizon, difficulty, seed):
    rng = random.Random(seed)

    lo = 10 ** (difficulty - 1)
    hi = 10 ** difficulty - 1  

    value = rng.randint(lo, hi)
    lines = [f"Start with {value}."]
    checkpoints = []

    for i in range(horizon):
        op = rng.choice(["add", "subtract", "multiply", "divide"])

        if op == "divide":
            # only divide by numbers that go in evenly, so value stays an integer
            divisors = [d for d in range(2, 11) if value % d == 0]
            if divisors:
                k = rng.choice(divisors)
                value = value // k
                lines.append(f"Step {i+1}: divide by {k}.")
                checkpoints.append(value)
                continue
            op = "add"  # nothing divides evenly, so use a different operation

        if op == "add":
            k = rng.randint(lo, hi)
            value = value + k
            lines.append(f"Step {i+1}: add {k}.")
        elif op == "subtract":
            k = rng.randint(lo, hi)
            value = value - k
            lines.append(f"Step {i+1}: subtract {k}.")
        elif op == "multiply":
            k = rng.randint(2, 9)
            value = value * k
            lines.append(f"Step {i+1}: multiply by {k}.")
        checkpoints.append(value)

    prompt = (
        "Follow the steps in order and report the final value.\n"
        + "\n".join(lines)
        + "\nEnd your response with a line of the form: ANSWER: <integer>"
    )
    return Task(
        id=f"h{horizon}-d{difficulty}-s{seed}",
        prompt=prompt,
        expected_answer=value,
        checkpoints=checkpoints,
        allowed_tools=set(),
        knobs={"horizon": horizon, "difficulty": difficulty, "seed": seed},
    )
