import json
import random

from activation_tracker import ActivationTracker
from utils import print_colored, extract_conversation, date_str
from utils_log import log_conversation, save_conversation_hidden_states
from system_agent import SystemAgent
from user_agent import UserAgent
from tasks import get_task
from model_huggingface import generate

# ──────────────────────────────────────────────
# HuggingFace generate() — drop-in replacement for model_openai.generate()
# Supports two backends:
#   1. LOCAL  – transformers pipeline on GPU (default, recommended for V100)
#   2. API    – HuggingFace Inference API (set HF_BACKEND=api or pass backend="api")
# ──────────────────────────────────────────────

import os
import time
from typing import Optional

HF_BACKEND = os.environ.get("HF_BACKEND", "local")   # "local" | "api"

# ──────────────────────────────────────────────
# ConversationSimulatorSharded (HF version)
# Identical logic to the original; only the import differs.
# ──────────────────────────────────────────────

class ConversationSimulatorSharded:
    def __init__(
        self,
        sample,
        assistant_model="meta-llama/Llama-3.1-8B-Instruct",
        system_model="meta-llama/Llama-3.1-8B-Instruct",
        user_model="meta-llama/Llama-3.1-8B-Instruct",
        assistant_temperature=0,
        user_temperature=0,
        dataset_fn=None,
        log_folder="logs",
        track_activation=False,
    ):
        self.task_name = sample["task"]
        self.task = get_task(self.task_name)
        self.dataset_fn = dataset_fn
        self.sample = sample
        self.system_model = system_model
        self.user_model = user_model
        self.user_agent = UserAgent(self.task, user_model)
        self.assistant_model = assistant_model
        self.system_agent = SystemAgent(self.task_name, system_model, self.sample)
        self.log_folder = log_folder
        self.system_message = self.task.generate_system_prompt(self.sample)
        self.answer_description = self.task.get_answer_description()

        self.run_with_custom_temperature = assistant_temperature != 1.0 or user_temperature != 1.0
        self.assistant_temperature = assistant_temperature
        self.user_temperature = user_temperature

        self.trace = [{"role": "system", "content": self.system_message, "timestamp": date_str()}]
        # self.trace = [{"role": "system", "content": "You are a helpful assistant.", "timestamp": date_str()}]
        
        self.activation_tracker = ActivationTracker(layers=[10, 20, 30, 40, 46], task=self.task, sample=self.sample["task_id"], track_full_hidden_states=True) if track_activation else None

    def get_num_turns(self, participant="assistant"):
        return sum(1 for msg in self.trace if msg["role"] == participant)

    def run(self, verbose=False, save_log=True):
        # Reasoning models (if using a local reasoning model, list its name below)
        # REASONING_MODEL_KEYWORDS = ("o1", "o3", "deepseek-r1", "qwq", "deepseek-r")
        # is_reasoning_model = any(kw in self.assistant_model.lower() for kw in REASONING_MODEL_KEYWORDS)
        # max_assistant_tokens = 10000 if is_reasoning_model else 1000
        
        # TODO
        max_assistant_tokens = 512 # TEMP: set to 512 for testing; can increase to 1000+ for final runs depending on model capacity

        is_completed, is_correct, score = False, False, None
        shards = self.sample["shards"]

        while not is_completed:         
            revealed_shard_ids = set(
                [msg["content"]["shard_id"] for msg in self.trace
                 if msg["role"] == "log" and msg["content"]["type"] == "shard_revealed"]
            )
            all_shards_revealed = len(revealed_shard_ids) == len(shards)
            if all_shards_revealed:
                if verbose:
                    print_colored(f"[log] all shards revealed ({revealed_shard_ids} / {len(shards)})", "blue")
                break
            
            is_first_turn = self.get_num_turns("assistant") == 0
            is_last_turn = len(revealed_shard_ids) == len(shards) - 1

            # 1. User response
            user_response, shard_revealed_id, cost_usd = self.user_agent.generate_response(
                self.trace, self.sample, temperature=self.user_temperature
            )
            
            self.trace.append({"role": "user", "content": user_response, "timestamp": date_str(), "cost_usd": cost_usd})
            if verbose:
                print_colored(f"[user] {user_response}", "green")

            if shard_revealed_id != -1:
                self.trace.append({"role": "log", "content": {"type": "shard_revealed", "shard_id": shard_revealed_id}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] shard revealed: {shard_revealed_id}", "blue")

            # 2. Assistant response  ← uses HF generate() instead of OpenAI
            assistant_response_obj = generate(
                messages=extract_conversation(self.trace, to_str=False),
                model_name=self.assistant_model,
                temperature=self.assistant_temperature,
                max_tokens=max_assistant_tokens,
                is_first_turn=is_first_turn,
                activation_tracker=self.activation_tracker,
                return_metadata=True
            )
            
            assistant_response = assistant_response_obj["message"]
            self.trace.append({"role": "assistant", "content": assistant_response, "timestamp": date_str(), "cost_usd": assistant_response_obj["total_usd"]})
            if verbose:
                print_colored(f"[assistant] {assistant_response}", "red")

            # 3. System verification
            system_verification_response, verification_cost_usd = self.system_agent.verify_system_response(self.trace)
            self.trace.append({"role": "log", "content": {"type": "system-verification", "response": system_verification_response}, "timestamp": date_str(), "cost_usd": verification_cost_usd})
            if verbose:
                print_colored(f"[log] system verification: {system_verification_response}", "blue")

            if system_verification_response["response_type"] == "answer_attempt":
                # 4. Evaluate
                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None

                if self.task_name == "summary" and not is_last_turn:
                    evaluation_return = {"score": 0.0}
                    score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)
                    
                    assert type(evaluation_return) is dict and ("score" in evaluation_return or "is_correct" in evaluation_return)
                    is_correct = evaluation_return.get("is_correct", None)
                    score = evaluation_return.get("score", None)

                if score == 1.0 and not is_correct:
                    is_correct = True

                self.trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": extracted_answer, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] answer evaluation:\n```{extracted_answer}\n```\n({'correct' if is_correct else 'incorrect'}; score: {score})", "blue")

                if is_correct:
                    is_completed = True
                    self.trace.append({"role": "log", "content": {"type": "conversation-completed", "is_correct": is_correct}, "timestamp": date_str()})
                    if verbose:
                        print_colored(f"[log] conversation completed: {is_correct}; score: {score}", "blue")

            elif system_verification_response["response_type"] in ["clarification", "discussion"]:
                continue
            
        if save_log:
            conv_type = "sharded"
            if self.run_with_custom_temperature:
                conv_type = f"sharded-at{self.assistant_temperature}-ut{self.user_temperature}"
            activation_result = self.activation_tracker.generate_result() if self.activation_tracker else None
            conv_id = log_conversation(
                conv_type, self.task.get_task_name(), self.sample["task_id"],
                self.dataset_fn, self.assistant_model, self.system_model,
                self.user_model, self.trace, activation_result, is_correct, score,
                log_folder=self.log_folder,
            )
            # 同步保存 hidden states
            if self.activation_tracker:
                save_conversation_hidden_states(
                    conv_id, conv_type, self.task.get_task_name(),
                    self.assistant_model, self.activation_tracker,
                    log_folder=self.log_folder
                )
        return is_correct, score


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ConversationSimulatorSharded with HuggingFace models")
    parser.add_argument("--task", type=str, default="math")
    parser.add_argument("--assistant_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HuggingFace model ID for the assistant")
    parser.add_argument("--system_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HuggingFace model ID for the system/verifier agent")
    parser.add_argument("--user_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HuggingFace model ID for the user agent")
    parser.add_argument("--backend", type=str, default=None, choices=["local", "api"],
                        help="HF backend: 'local' (transformers) or 'api' (HF Inference API). "
                             "Overrides HF_BACKEND env var.")
    parser.add_argument("--dataset", type=str, default="data/sharded_instructions_600.json")
    
    parser.add_argument("--track_activation", default=True, help="Whether to track activations (for local models only)")
    
    args = parser.parse_args()

    # Set backend via env var so UserAgent / SystemAgent also pick it up if they call generate()
    if args.backend:
        os.environ["HF_BACKEND"] = args.backend

    with open(args.dataset, "r") as f:
        data = json.load(f)

    # data = [d for d in data if d["task"] == args.task]
    data = [d for d in data if d["task"] == args.task]
    
    for i in range(15):
        count = [d for d in data if len(d["shards"]) == i]  # TEMP: only test on 3-shard tasks for now
        print(f"Tasks with {i} shards: {len(count)}")

    # sample = random.choice(data)
    # print(f"[main] Running task='{args.task}', sample id='{sample.get('task_id', '?')}'")

    # simulator = ConversationSimulatorSharded(
    #     sample=sample,
    #     assistant_model=args.assistant_model,
    #     system_model=args.system_model,
    #     user_model=args.user_model,
    #     dataset_fn=args.dataset,
    #     hf_backend=args.backend,
    #     track_activation=args.track_activation,
    # )
    # is_correct, score = simulator.run(verbose=True, save_log=True)
