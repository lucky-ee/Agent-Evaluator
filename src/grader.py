from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Input types (match these to your generator / runner)
# ---------------------------------------------------------------------------


@dataclass
class Task:
    id: str
    prompt: str
    expected_answer: Any                      # str, int, float, or "a/b"
    checkpoints: list[Any] = field(default_factory=list)  # intermediate values, in order
    allowed_tools: set[str] = field(default_factory=set)
    knobs: dict[str, Any] = field(default_factory=dict)   # horizon, difficulty, ...


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None


@dataclass
class Step:
    content: str
    tool_call: ToolCall | None = None


Termination = Literal["finished", "step_limit", "context_overflow", "error"]


@dataclass
class Trajectory:
    task_id: str
    steps: list[Step]
    final_output: str
    termination: Termination

# Output types

class Outcome(Enum):
    CORRECT = "correct"
    PARTIAL = "partial"
    WRONG = "wrong"


class Failure(Enum):
    STEP_LIMIT = "step_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    RUNTIME_ERROR = "runtime_error"
    MALFORMED = "malformed"
    HALLUCINATED_TOOL = "hallucinated_tool"
    WRONG_ANSWER = "wrong_answer"


@dataclass
class GradeResult:
    task_id: str
    outcome: Outcome
    score: float
    failure: Failure | None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flat dict for JSONL logging."""
        return {
            "task_id": self.task_id,
            "outcome": self.outcome.value,
            "score": self.score,
            "failure": self.failure.value if self.failure else None,
            "details": self.details,
        }

# Answer parsing and comparison

_LABELED_ANSWER_RE = re.compile(r"(?:final\s*answer|answer)\s*[:=]\s*(.+)", re.IGNORECASE)
_BOXED_START_RE = re.compile(r"\\boxed\{")

# A '-' only counts as a sign when it isn't glued to a preceding digit/decimal
# point, e.g. "10-5=5" should tokenize as 10, 5, 5, not 10, -5, 5.
NUMBER_RE = re.compile(r"(?<![\d.])-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?(?:/\d+)?")

REL_TOL = 1e-6
ABS_TOL = 1e-9


def _find_boxed_contents(text: str) -> list[tuple[int, str]]:
    """Find \\boxed{...} occurrences, matching braces up to arbitrary nesting
    depth (plain regex can't express that)."""
    found = []
    for start_match in _BOXED_START_RE.finditer(text):
        depth = 1
        i = start_match.end()
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            found.append((start_match.start(), text[start_match.end():i - 1]))
    return found


def parse_answer(final_output: str) -> str | None:
    if not final_output:
        return None
    candidates: list[tuple[int, str]] = [
        (m.start(), m.group(1)) for m in _LABELED_ANSWER_RE.finditer(final_output)
    ]
    candidates.extend(_find_boxed_contents(final_output))
    if not candidates:
        return None
    # The model's actual final answer is whichever candidate appears last in
    # the text, not whichever pattern happens to be listed first.
    candidates.sort(key=lambda c: c[0])
    ans = candidates[-1][1].strip().rstrip(".").strip()
    return ans or None


def to_number(value: Any) -> Fraction | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Fraction):
        return value
    if not isinstance(value, str):
        return None
    s = value.strip().replace(",", "").replace(" ", "").rstrip(".")
    s = s.replace("$", "")
    if not s:
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def normalize_text(value: Any) -> str:
    s = str(value).strip().lower().rstrip(".")
    s = s.replace("$", "")
    return re.sub(r"\s+", " ", s)


def answer_matches(got: Any, expected: Any) -> bool:
    if got is None or expected is None:
        return False
    g, e = to_number(got), to_number(expected)
    if g is not None and e is not None:
        return math.isclose(float(g), float(e), rel_tol=REL_TOL, abs_tol=ABS_TOL)
    got_text, expected_text = normalize_text(got), normalize_text(expected)
    if not got_text or not expected_text:
        return False
    return got_text == expected_text

# Partial credit

def _step_texts(traj: Trajectory) -> list[str]:
    texts = []
    for step in traj.steps:
        parts = [step.content or ""]
        if step.tool_call is not None and step.tool_call.result is not None:
            parts.append(str(step.tool_call.result))
        texts.append(" ".join(parts))
    return texts


def _text_contains(text: str, value: Any) -> bool:
    target = to_number(value)
    if target is not None:
        return any(answer_matches(tok, target) for tok in NUMBER_RE.findall(text))
    needle = re.escape(normalize_text(value))
    return re.search(rf"\b{needle}\b", normalize_text(text)) is not None


def partial_credit(
    traj: Trajectory,
    checkpoints: list[Any],
    mode: Literal["prefix", "fraction"] = "prefix",
) -> tuple[float, int]:
    if not checkpoints:
        return 0.0, 0
    texts = _step_texts(traj)

    if mode == "fraction":
        full = " ".join(texts)
        hit = sum(_text_contains(full, cp) for cp in checkpoints)
        return hit / len(checkpoints), hit

    hit, step_idx = 0, 0
    for cp in checkpoints:
        while step_idx < len(texts) and not _text_contains(texts[step_idx], cp):
            step_idx += 1
        if step_idx == len(texts):
            break
        hit += 1
        step_idx += 1  # next checkpoint must show up in a later step, not this same one
    return hit / len(checkpoints), hit

# Tool hallucination

def find_hallucinated_tools(traj: Trajectory, allowed: set[str]) -> list[str]:
    """Names of tools called that aren't in the task's allowed set.
    (Stronger check, left for later: model claims a tool returned something
    that doesn't match the logged result.)"""
    seen: list[str] = []
    for step in traj.steps:
        tc = step.tool_call
        if tc is not None and tc.name not in allowed and tc.name not in seen:
            seen.append(tc.name)
    return seen

# Dispatcher

_TERMINATION_FAILURE = {
    "step_limit": Failure.STEP_LIMIT,
    "context_overflow": Failure.CONTEXT_OVERFLOW,
    "error": Failure.RUNTIME_ERROR,
}


def _fail(
    task: Task,
    traj: Trajectory,
    failure: Failure,
    credit_mode: Literal["prefix", "fraction"],
    **details: Any,
) -> GradeResult:
    score, hit = partial_credit(traj, task.checkpoints, credit_mode)
    # A failed run never gets full credit, even if every checkpoint was hit.
    score = min(score, 1.0 - 1e-9) if task.checkpoints else 0.0
    outcome = Outcome.PARTIAL if score > 0 else Outcome.WRONG
    details.update(checkpoints_hit=hit, checkpoints_total=len(task.checkpoints))
    return GradeResult(task.id, outcome, score, failure, details)


def grade(
    task: Task,
    traj: Trajectory,
    credit_mode: Literal["prefix", "fraction"] = "prefix",
) -> GradeResult:
    if traj.task_id != task.id:
        raise ValueError(f"trajectory {traj.task_id!r} does not belong to task {task.id!r}")

    # 1. Did the run finish?
    if traj.termination != "finished":
        try:
            failure = _TERMINATION_FAILURE[traj.termination]
        except KeyError:
            raise ValueError(f"unknown termination value: {traj.termination!r}") from None
        return _fail(task, traj, failure, credit_mode, termination=traj.termination)

    # 2. Did it call tools that don't exist?
    bad_tools = find_hallucinated_tools(traj, task.allowed_tools)
    if bad_tools:
        return _fail(task, traj, Failure.HALLUCINATED_TOOL, credit_mode, bad_tools=bad_tools)

    # 3. Can we extract an answer?
    answer = parse_answer(traj.final_output)
    if answer is None:
        return _fail(task, traj, Failure.MALFORMED, credit_mode)

    # 4. Is it right?
    if answer_matches(answer, task.expected_answer):
        return GradeResult(
            task.id, Outcome.CORRECT, 1.0, None,
            {
                "answer": answer,
                "checkpoints_total": len(task.checkpoints),
                "checkpoints_hit": len(task.checkpoints),
            },
        )

    return _fail(task, traj, Failure.WRONG_ANSWER, credit_mode, got=answer)