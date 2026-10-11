import random


def generate_problem(horizon, difficulty, seed):
    rng = random.Random(seed)

    lo = 10 ** (difficulty - 1)
    hi = 10 ** difficulty - 1  

    value = rng.randint(lo, hi)
    lines = [f"Start with {value}."]

    for i in range(horizon):
        op = rng.choice(["add", "subtract", "multiply", "divide"])

        if op == "divide":
            # only divide by numbers that go in evenly, so value stays an integer
            divisors = [d for d in range(2, 11) if value % d == 0]
            if divisors:
                k = rng.choice(divisors)
                value = value // k
                lines.append(f"Step {i+1}: divide by {k}.")
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

    prompt = (
        "Follow the steps in order and report the final value.\n"
        + "\n".join(lines)
        + "\nEnd your response with a line of the form: ANSWER: <integer>"
    )
    return prompt, value


if __name__ == "__main__":
    p1, a1 = generate_problem(4, 2, seed=0)
    p2, a2 = generate_problem(4, 2, seed=0)
    assert (p1, a1) == (p2, a2), "same seed must give same problem"

    p3, a3 = generate_problem(4, 2, seed=1)
    assert p3 != p1, "different seeds should differ"

    # independent check: replay the steps from the text and compare to the answer
    import re
    def replay(prompt):
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

    for h in (1, 4, 8, 20):
        for d in (1, 2, 3):
            for s in range(200):
                p, a = generate_problem(h, d, s)
                assert replay(p) == a, (h, d, s)
                assert " to get " not in p, "prompt must not print running totals"
    print("all checks passed\n")
    print(p1)
    print("answer:", a1)