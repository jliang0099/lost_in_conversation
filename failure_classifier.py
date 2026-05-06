"""
failure_classifier.py

Classifies why tasks failed (score=0 or score=None) in Lost-in-Conversation JSONL log files.

Supported task types: code, actions, math
Usage:
    python failure_classifier.py <path_to_jsonl> [--task-type auto|code|actions|math]
    python failure_classifier.py logs/code/.../*.jsonl --task-type code
"""

import json
import sys
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _get_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content or "")


def _load_entries(path: str) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _parse_log_content(msg: dict) -> Optional[dict]:
    """Return the parsed dict payload of a log message, or None."""
    if msg.get("role") != "log":
        return None
    c = msg.get("content", {})
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except json.JSONDecodeError:
            return None
    return c if isinstance(c, dict) else None


def _extract_trace_events(trace: list[dict]) -> dict:
    """
    Walk a trace and extract structured events into named lists.
    Returns dict with keys:
      shards, verifications, evaluations, user_msgs, assistant_msgs
    """
    shards = []
    verifications = []
    evaluations = []
    user_msgs = []
    assistant_msgs = []

    for msg in trace:
        role = msg.get("role", "")
        c = _parse_log_content(msg)

        if c is not None:
            t = c.get("type", "")
            if t == "shard_revealed":
                shards.append(c)
            elif t == "system-verification":
                verifications.append(c.get("response", {}))
            elif t == "answer-evaluation":
                evaluations.append(c)
        elif role == "user":
            user_msgs.append(_get_text(msg.get("content", "")))
        elif role == "assistant":
            assistant_msgs.append(_get_text(msg.get("content", "")))

    return dict(
        shards=shards,
        verifications=verifications,
        evaluations=evaluations,
        user_msgs=user_msgs,
        assistant_msgs=assistant_msgs,
    )


# ---------------------------------------------------------------------------
# Task-type detection
# ---------------------------------------------------------------------------

def _detect_task_type(entries: list[dict]) -> str:
    """Infer task type from conv_type field or system prompt content."""
    for e in entries[:20]:
        ct = e.get("conv_type") or e.get("task", "")
        if ct:
            ct = ct.lower()
            if "code" in ct:
                return "code"
            if "action" in ct or "function" in ct:
                return "actions"
            if "math" in ct:
                return "math"
        # Fallback: peek at system prompt
        trace = e.get("trace", [])
        if trace:
            sys_msg = _get_text(trace[0].get("content", ""))
            if "python programmer" in sys_msg.lower():
                return "code"
            if "composing functions" in sys_msg.lower():
                return "actions"
            if "mathematical" in sys_msg.lower() or "expert problem solver" in sys_msg.lower():
                return "math"
    return "unknown"


# ---------------------------------------------------------------------------
# Failure category definitions
# ---------------------------------------------------------------------------

# Each classifier returns a string category label.

CODE_CATEGORIES = {
    "ignores_all_hints": (
        "Model produces the same wrong answer across all shards, "
        "ignoring every clarifying hint."
    ),
    "converges_wrong": (
        "Model partially updates its answer with each hint but settles on "
        "an incorrect solution before the last shard."
    ),
    "evolves_wrong_final": (
        "Answer keeps changing through all shards but the final submission "
        "is still incorrect."
    ),
    "no_answer_attempt": (
        "Model never submitted an evaluatable answer."
    ),
}

ACTIONS_CATEGORIES = {
    "wrong_function_count": (
        "Model called the wrong number of functions (e.g., one call when "
        "multiple are required, or vice versa)."
    ),
    "no_matching_function": (
        "Model's function call(s) don't match any expected answer "
        "(wrong function name or unresolvable output)."
    ),
    "wrong_param_value": (
        "Correct function(s) called but one or more parameter values are wrong."
    ),
    "wrong_param_type": (
        "Correct function called but a parameter has the wrong data type."
    ),
    "missing_optional_param": (
        "Model omitted an optional parameter that was required by the ground truth."
    ),
    "no_answer_attempt": (
        "Model never submitted an evaluatable function call."
    ),
}

MATH_CATEGORIES = {
    "same_wrong_despite_hints": (
        "Model calculates a wrong answer early and sticks to it regardless of "
        "subsequent clarifying hints."
    ),
    "evolves_stays_wrong": (
        "Model updates its calculation with each hint but never reaches the "
        "correct answer (formula / arithmetic error)."
    ),
    "attempted_but_unresolved": (
        "Model attempted to submit answer (1-4 evaluations) but conversation "
        "ended in discussion state, never finalizing to a confirmed result."
    ),
    "never_attempted_answer": (
        "Model never entered answer submission flow (0 evaluations). Stayed in "
        "discussion/clarification/refusal state throughout, never tried to commit."
    ),
    "refuses_to_answer": (
        "Model explicitly refuses or goes silent when given further information."
    ),
    "no_answer_attempt": (
        "Model never submitted an evaluatable answer."
    ),
}

ALL_CATEGORY_DESCRIPTIONS = {
    **CODE_CATEGORIES,
    **ACTIONS_CATEGORIES,
    **MATH_CATEGORIES,
}


# ---------------------------------------------------------------------------
# Per-type classifiers
# ---------------------------------------------------------------------------

def _classify_code(events: dict) -> str:
    evals = events["evaluations"]
    if not evals:
        return "no_answer_attempt"

    answers = [e.get("exact_answer", "") for e in evals]
    unique_answers = list(dict.fromkeys(answers))  # ordered dedup

    if len(unique_answers) == 1:
        return "ignores_all_hints"

    # Check if the model stopped changing at some point before the last shard
    last_idx = len(answers) - 1
    if answers[-1] != answers[-2]:
        return "evolves_wrong_final"
    return "converges_wrong"


def _classify_actions(events: dict) -> str:
    evals = events["evaluations"]
    if not evals:
        return "no_answer_attempt"

    last_eval = evals[-1]
    eval_ret = last_eval.get("evaluation_return", {})
    errors = eval_ret.get("error", [])
    if not isinstance(errors, list):
        errors = [errors] if errors else []

    for err in errors:
        err_str = str(err)
        if "Wrong number of functions" in err_str:
            return "wrong_function_count"
        if "Could not find a matching function" in err_str:
            return "no_matching_function"
        if isinstance(err, dict):
            sub_type = err.get("sub_error_type", "")
        else:
            sub_type = err_str
        if "type_error" in sub_type:
            return "wrong_param_type"
        if "missing_optional" in sub_type:
            return "missing_optional_param"
        if "value_error" in sub_type:
            return "wrong_param_value"

    # No structured error field - check is_correct
    if not last_eval.get("is_correct", True):
        return "no_matching_function"

    return "no_answer_attempt"


def _classify_math(events: dict, entry: dict) -> str:
    evals = events["evaluations"]
    verifications = events["verifications"]

    # No score = never entered answer submission flow
    if entry.get("score") is None:
        return "never_attempted_answer"

    # score=0 but has evaluations: check if ended in discussion
    if not evals:
        return "no_answer_attempt"

    last_resp_type = verifications[-1].get("response_type", "") if verifications else ""

    if last_resp_type == "refuse":
        return "refuses_to_answer"

    # Model tried (has evals) but ended in discussion
    if last_resp_type in ("discussion", "clarification", "interrogation"):
        return "attempted_but_unresolved"

    answers = [e.get("exact_answer", "") for e in evals]
    unique_answers = list(dict.fromkeys(answers))

    if len(unique_answers) == 1:
        return "same_wrong_despite_hints"
    return "evolves_stays_wrong"


# ---------------------------------------------------------------------------
# Main classifier dispatch
# ---------------------------------------------------------------------------

@dataclass
class FailureResult:
    conv_id: str
    category: str
    task_type: str
    num_shards: int
    num_eval_attempts: int
    last_answer: str = ""
    last_error: str = ""


def classify_entry(entry: dict, task_type: str) -> Optional[FailureResult]:
    """Return a FailureResult if the entry failed, else None."""
    score = entry.get("score")
    # Treat both score=0 and score=None as failure
    if score not in (0, 0.0, None):
        return None

    trace = entry.get("trace", [])
    events = _extract_trace_events(trace)

    if task_type == "code":
        category = _classify_code(events)
    elif task_type == "actions":
        category = _classify_actions(events)
    elif task_type == "math":
        category = _classify_math(events, entry)
    else:
        category = "unknown_task_type"

    evals = events["evaluations"]
    last_answer = evals[-1].get("exact_answer", "") if evals else ""
    last_error = ""
    if evals:
        eval_ret = evals[-1].get("evaluation_return", {})
        errors = eval_ret.get("error", [])
        if isinstance(errors, list) and errors:
            last_error = str(errors[0])[:120]

    return FailureResult(
        conv_id=entry.get("conv_id", ""),
        category=category,
        task_type=task_type,
        num_shards=len(events["shards"]),
        num_eval_attempts=len(evals),
        last_answer=str(last_answer)[:120],
        last_error=last_error,
    )


# ---------------------------------------------------------------------------
# Analysis & reporting
# ---------------------------------------------------------------------------

def analyze_file(path: str, task_type: str = "auto", verbose: bool = False) -> dict:
    """
    Analyze a JSONL log file and return a summary dict.

    Returns:
        {
          "path": str,
          "task_type": str,
          "total": int,
          "failed": int,
          "pass_rate": float,
          "category_counts": Counter,
          "failures": list[FailureResult],
        }
    """
    entries = _load_entries(path)

    if task_type == "auto":
        task_type = _detect_task_type(entries)

    failures = []
    for e in entries:
        result = classify_entry(e, task_type)
        if result is not None:
            failures.append(result)

    counts = Counter(r.category for r in failures)
    total = len(entries)
    failed = len(failures)

    return dict(
        path=path,
        task_type=task_type,
        total=total,
        failed=failed,
        pass_rate=(total - failed) / total if total else 0.0,
        category_counts=counts,
        failures=failures,
    )


def print_report(summary: dict, verbose: bool = False) -> None:
    path = summary["path"]
    task_type = summary["task_type"]
    total = summary["total"]
    failed = summary["failed"]
    passed = total - failed
    pass_rate = summary["pass_rate"]
    counts = summary["category_counts"]
    failures = summary["failures"]

    print(f"\n{'='*70}")
    print(f"File      : {Path(path).name}")
    print(f"Task type : {task_type}")
    print(f"Total     : {total}  |  Passed: {passed}  |  Failed: {failed}  |  Pass rate: {pass_rate:.1%}")
    print(f"\nFailure breakdown ({failed} cases):")

    descriptions = CODE_CATEGORIES if task_type == "code" else (
        ACTIONS_CATEGORIES if task_type == "actions" else MATH_CATEGORIES
    )

    for cat, cnt in counts.most_common():
        pct = cnt / failed * 100 if failed else 0
        desc = descriptions.get(cat, ALL_CATEGORY_DESCRIPTIONS.get(cat, ""))
        print(f"  {cnt:4d} ({pct:5.1f}%)  [{cat}]")
        if desc:
            print(f"             {desc}")

    if verbose and failures:
        print(f"\nSample failures (up to 3 per category):")
        per_cat: dict[str, list] = defaultdict(list)
        for f in failures:
            per_cat[f.category].append(f)
        for cat, items in per_cat.items():
            print(f"\n  Category: {cat}")
            for item in items[:3]:
                print(f"    conv_id={item.conv_id}  shards={item.num_shards}  attempts={item.num_eval_attempts}")
                if item.last_answer:
                    print(f"    last_answer: {item.last_answer[:80]}")
                if item.last_error:
                    print(f"    last_error:  {item.last_error[:80]}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classify failure reasons in Lost-in-Conversation JSONL logs."
    )
    parser.add_argument("paths", nargs="+", help="One or more .jsonl log files to analyze.")
    parser.add_argument(
        "--task-type",
        default="auto",
        choices=["auto", "code", "actions", "math"],
        help="Task type to use for classification (default: auto-detect).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show sample failures per category.",
    )
    parser.add_argument(
        "--json-out", metavar="FILE",
        help="Write full results as JSON to this file.",
    )

    args = parser.parse_args()

    all_summaries = []
    for path in args.paths:
        summary = analyze_file(path, task_type=args.task_type, verbose=args.verbose)
        print_report(summary, verbose=args.verbose)
        all_summaries.append(summary)

    if args.json_out:
        # Convert Counters to plain dicts for JSON serialization
        serializable = []
        for s in all_summaries:
            serializable.append({
                **{k: v for k, v in s.items() if k not in ("category_counts", "failures")},
                "category_counts": dict(s["category_counts"]),
                "failures": [
                    {
                        "conv_id": f.conv_id,
                        "category": f.category,
                        "num_shards": f.num_shards,
                        "num_eval_attempts": f.num_eval_attempts,
                        "last_answer": f.last_answer,
                        "last_error": f.last_error,
                    }
                    for f in s["failures"]
                ],
            })
        with open(args.json_out, "w", encoding="utf-8") as fp:
            json.dump(serializable, fp, indent=2, ensure_ascii=False)
        print(f"Results written to {args.json_out}")


if __name__ == "__main__":
    main()
