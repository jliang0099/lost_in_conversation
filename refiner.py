"""
Refiner — Step 1 of the Mediator-Assistant framework.

Reads failed sharded conversations (D-) and their corresponding full
questions (D+) from the dataset, then asks an LLM to distill them into
a compact set of rewriting rules (experiences) that will later guide
the Mediator.

Usage:
    python refiner.py \
        --log_file logs/math/sharded-at0-ut0/"(40-103)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl" \
        --dataset_file data/sharded_instructions_600.json \
        --task math \
        --n_examples 5 \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output experiences/math.json
"""

import argparse
import json
import os
import random

from utils import extract_conversation
from model_huggingface import generate

REFINER_PROMPT = """\
You are a meta-analysis AI assistant responsible for extracting effective \
rewriting rules and best practices from examples of Conversation Transcripts \
between a User and an AI assistant, together with the Ground Truth rewritten request.

Your goal is to produce a clear, actionable set of RULES that can guide another \
"Request Rewriting Mediator" to accurately transform a given Conversation Transcript \
into a clear, first-person user request.

The RULES you produce must be:
- Few in number, focusing only on the most critical factors that determine correct rewriting.
- Direct, actionable, and applicable to any future Conversation Transcript.

[Input You Will Receive]
A list of examples, each containing:
1. Conversation Transcript (dialogue between the User and the AI assistant).
2. Ground Truth Rewritten Request (ideal output for that conversation).

[Your Tasks]
1. For each example:
   - Compare the Conversation Transcript with its Ground Truth request.
   - Note what transformations were applied (e.g., summarization, rephrasing, \
deletion of irrelevant parts, conversion to first-person, clarification of \
ambiguous points, tone adjustment).
2. Identify common patterns and shared decision rules across all examples.
3. Merge these insights into one unified, concise set of numbered RULES:
   - RULES must work for *new, unseen conversations*.
   - RULES must be clear, direct, and applicable without additional explanation.

[Output Format]
Output only the RULES section as:
[[RULES]]
1. ...
2. ...
3. ...
[[END]]

Do not include any commentary outside the [[RULES]] section.

[Now process the following data]
[[CONVERSATIONS]]"""


def format_conversation(trace: list[dict]) -> str:
    """Format a trace into a readable transcript (user/assistant turns only)."""
    lines = []
    for msg in trace:
        if msg["role"] == "user":
            lines.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            lines.append(f"Assistant: {msg['content']}")
    return "\n".join(lines)


def build_conversations_block(pairs: list[dict]) -> str:
    """Build the [[CONVERSATIONS]] section from a list of (D-, D+) pairs."""
    blocks = []
    for i, pair in enumerate(pairs, 1):
        block = (
            f"--- Example {i} ---\n"
            f"Conversation Transcript:\n{pair['transcript']}\n\n"
            f"Ground Truth Rewritten Request:\n{pair['full_question']}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def extract_rules(raw_output: str) -> str:
    """Pull the text between [[RULES]] and [[END]]."""
    start = raw_output.find("[[RULES]]")
    end = raw_output.find("[[END]]")
    if start != -1 and end != -1:
        return raw_output[start + len("[[RULES]]"):end].strip()
    # Fallback: return everything if markers are missing
    return raw_output.strip()


def main(args):
    # 1. Load log file (D-)
    entries = []
    with open(args.log_file) as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    failed = [e for e in entries if not e.get("is_correct")]
    print(f"Loaded {len(entries)} conversations, {len(failed)} failed.")

    # 2. Load dataset (D+)
    with open(args.dataset_file) as f:
        dataset = json.load(f)
    task_to_question = {d["task_id"]: d for d in dataset if d["task"] == args.task}
    print(f"Loaded {len(task_to_question)} {args.task} samples from dataset.")

    # 3. Find failed instances that have a matching dataset entry with a full question
    matched = []
    for entry in failed:
        tid = entry["task_id"]
        sample = task_to_question.get(tid)
        if sample is None:
            continue
        # D+ is the full question field (math uses 'question', others may use 'prompt')
        full_question = sample.get("question") or sample.get("prompt") or ""
        if not full_question:
            continue
        matched.append({"entry": entry, "full_question": full_question})

    print(f"Matched {len(matched)} failed entries to dataset questions.")

    if len(matched) < args.n_examples:
        print(f"Warning: only {len(matched)} matched pairs available (requested {args.n_examples}).")

    random.seed(42)
    selected = random.sample(matched, min(args.n_examples, len(matched)))

    # 4. Build contrastive pairs
    pairs = []
    for item in selected:
        trace = item["entry"]["trace"]
        transcript = format_conversation(trace)
        pairs.append({
            "task_id": item["entry"]["task_id"],
            "transcript": transcript,
            "full_question": item["full_question"],
        })

    # 5. Build the prompt
    conversations_block = build_conversations_block(pairs)
    prompt = REFINER_PROMPT.replace("[[CONVERSATIONS]]", conversations_block)

    print(f"\nSending {len(pairs)} contrastive pairs to {args.model} ...")
    print("-" * 60)

    result = generate(
        messages=[{"role": "user", "content": prompt}],
        model_name=args.model,
        temperature=0.0,
        max_tokens=8192,
        return_metadata=True,
    )
    raw_output = result["message"]

    print("Raw LLM output:")
    print(raw_output)
    print("-" * 60)

    rules = extract_rules(raw_output)
    print("Extracted rules:")
    print(rules)

    # 6. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "task": args.task,
        "model": args.model,
        "n_examples": len(pairs),
        "source_log": args.log_file,
        "rules": rules,
        "raw_output": raw_output,
        "pairs_used": [{"task_id": p["task_id"]} for p in pairs],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved experiences to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True,
                        help="Path to the sharded JSONL log file (D-)")
    parser.add_argument("--dataset_file", type=str,
                        default="data/sharded_instructions_600.json",
                        help="Path to the dataset JSON (D+)")
    parser.add_argument("--task", type=str, default="math",
                        help="Task name (math / code / database / actions)")
    parser.add_argument("--n_examples", type=int, default=5,
                        help="Number of contrastive pairs to use")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model to use for Refiner LLM call")
    parser.add_argument("--output", type=str, default="experiences/math.json",
                        help="Output path for the generated experiences")
    args = parser.parse_args()
    main(args)
