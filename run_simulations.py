import argparse
import random
import multiprocessing
import json
import tqdm
from huggingface_simulator_sharded import ConversationSimulatorSharded
from simulator_snowball import ConversationSimulatorSnowball
# from simulator_full import ConversationSimulatorFull
from concurrent.futures import ThreadPoolExecutor
from utils_log import get_run_counts
from collections import Counter
from lu_bandit.agent import LUBandit
from ce_methods import build_ce_method, CHOICES as CE_CHOICES

def run_simulation(todo):
    dataset_fn = todo["dataset_fn"]
    try:
        assistant_temp = todo.get("assistant_temperature", 0)
        user_temp = todo.get("user_temperature", 0)

        if "sharded" in todo["conv_type"]:
            conversation_simulator = ConversationSimulatorSharded(
                todo["sample"],
                assistant_model=todo["assistant_model"],
                system_model=todo["system_model"],
                user_model=todo["user_model"],
                assistant_temperature=assistant_temp,
                user_temperature=user_temp,
                dataset_fn=dataset_fn,
                log_folder=args.log_folder,
                track_activation=True,
                conv_type=todo["conv_type"],

                lu_bandit=todo.get("lu_bandit", None),
                ce_method=todo.get("ce_method", None),
            )
        elif todo["conv_type"].startswith("snowball"):
            conversation_simulator = ConversationSimulatorSnowball(
                todo["sample"],
                assistant_model=todo["assistant_model"],
                system_model=todo["system_model"],
                user_model=todo["user_model"],
                assistant_temperature=assistant_temp,
                user_temperature=user_temp,
                dataset_fn=dataset_fn,
                log_folder=args.log_folder,
                track_activation=True,
            )

        conversation_simulator.run(verbose=args.verbose)

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        tqdm.tqdm.write(f"\033[91m [Error on {todo['sample']['task_id']}; {todo['assistant_model']}; {todo['conv_type']}]:\n{error_msg}\033[0m")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_file", type=str, default="data/sharded_instructions_600.json", help="Dataset file to use")

    parser.add_argument("--N_full_runs", type=int, default=1, help="Number of full runs per model")
    parser.add_argument("--N_concat_runs", type=int, default=1, help="Number of concat runs per model")
    parser.add_argument("--N_sharded_runs", type=int, default=1, help="Number of sharded runs per model")
    parser.add_argument("--models", nargs="+", default=["meta-llama/Llama-3.1-8B-Instruct"], # `, "gpt-4o"`
                        help="List of models to run experiments with")
    parser.add_argument("--tasks", nargs="+", default=["math"], help="Tasks to run experiments with") # "code", "database", "actions", "math", "data2text", "summary", "translation"
    parser.add_argument("--system_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="System model to use")
    parser.add_argument("--user_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="User model to use")
    parser.add_argument("--N_workers", type=int, default=1, help="Number of workers to run experiments with")
    parser.add_argument("--log_folder", type=str, default="logs", help="Log folder to use")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    parser.add_argument("--assistant_temperature", type=float, default=0, help="Temperature to use for assistant models")
    parser.add_argument("--user_temperature", type=float, default=0, help="Temperature to use for user models")
    
    parser.add_argument("--failure_category", type=str, default=None,
                        help="Only run tasks from this failure category (e.g. same_wrong_despite_hints). "
                             "Requires --category_ids_file.")
    parser.add_argument("--category_ids_file", type=str, default="category_task_ids.json",
                        help="JSON file produced by: python failure_classifier.py ... --export-task-ids FILE")

    # ── CE Methods ────────────────────────────────────────────────────────────
    parser.add_argument("--ce_method", type=str, default="none", choices=CE_CHOICES,
                        help="Context Engineering method to apply (default: none)")
    
    parser.add_argument("--fewshot_k", type=int, default=3,
                        help="Number of few-shot demo examples (used when --ce_method fewshot)")
    parser.add_argument("--fewshot_log", type=str,
                        default="logs/math/sharded-at0-ut0/(260-428)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl",
                        help="Demo log file for FewShotCE (task_ids must not overlap with evaluation set)")
    
    parser.add_argument("--compress_model", type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help="Model used to generate the conversation summary "
                             "(context_compression). Defaults to --system_model.")
    parser.add_argument("--compress_keep_last_n", type=int, default=2,
                        help="Recent user/assistant turn pairs kept verbatim (context_compression)")
    parser.add_argument("--compress_min_history", type=int, default=2,
                        help="Minimum history turns before compression kicks in (context_compression)")
    parser.add_argument("--compress_max_tokens", type=int, default=256,
                        help="Token budget for the generated summary (context_compression)")

    # ── LUBandit ──────────────────────────────────────────────────────────────
    parser.add_argument("--lu_bandit", action="store_true", default=False,
                        help="Enable LUBandit adaptive compression policy")
    parser.add_argument("--bandit_alpha", type=float, default=1.0,
                        help="LinUCB exploration coefficient (default: 1.0)")
    parser.add_argument("--bandit_epsilon", type=float, default=0.1,
                        help="ε-greedy override on top of UCB (0 = pure UCB)")
    parser.add_argument("--bandit_lambda_drift", type=float, default=0.5,
                        help="Curvature-drift penalty weight λ_drift (default: 0.5)")
    parser.add_argument("--bandit_final_bonus", type=float, default=1.0,
                        help="Task-score bonus weight applied at end of episode (default: 1.0)")
    parser.add_argument("--bandit_checkpoint", type=str, default=None,
                        help="Path to save/load LinUCB weights (pkl). Loaded if exists, saved after each episode.")
    parser.add_argument("--bandit_discriminator_model", type=str, default=None,
                        help="Model for vLLM quality discriminator (defaults to --system_model)")
    parser.add_argument("--bandit_discriminator_mode", type=str, default="none",
                        choices=["none", "prompt", "logprob"],
                        help="Discriminator mode: 'none' (skip), 'prompt' (LLM scoring), 'logprob'")

    args = parser.parse_args()

    # Build CE method once; stateless, safe to share across threads.
    _ce_kwargs = {}
    if args.ce_method == "fewshot":
        _ce_kwargs = {"log_path": args.fewshot_log, "k": args.fewshot_k}
    elif args.ce_method == "context_compression":
        _ce_kwargs = {
            "model_name":        args.compress_model or args.system_model,
            "keep_last_n_turns": args.compress_keep_last_n,
            "min_history_turns": args.compress_min_history,
            "summary_max_tokens": args.compress_max_tokens,
        }
    ce_method = build_ce_method(args.ce_method, **_ce_kwargs)
    if ce_method is not None:
        print(f"[ce_method] {ce_method.name}")

    # Build LUBandit once; shared across worker threads (bandit weights are
    # updated inside each thread — use --N_workers 1 for correct online learning).
    lu_bandit = None
    if args.lu_bandit:
        discriminator_model = args.bandit_discriminator_model or args.system_model
        lu_bandit = LUBandit.from_config(
            alpha=args.bandit_alpha,
            epsilon=args.bandit_epsilon,
            lambda_drift=args.bandit_lambda_drift,
            final_bonus=args.bandit_final_bonus,
            discriminator_model=discriminator_model,
            discriminator_mode=args.bandit_discriminator_mode,
            checkpoint_path=args.bandit_checkpoint,
        )
        print(
            f"[lu_bandit] alpha={args.bandit_alpha}  epsilon={args.bandit_epsilon}  "
            f"actions=neutral/explore/commit/challenge  discriminator={args.bandit_discriminator_mode}"
        )

    # windows fix dataset_file to be unix format
    dataset_fn = args.dataset_file
    if args.dataset_file.startswith(".\\"):
        dataset_fn = args.dataset_file[2:]
    dataset_fn = dataset_fn.replace("\\", "/")

    with open(dataset_fn, "r") as f:
        samples = json.load(f)

    samples = [d for d in samples if d["task"] in args.tasks]

    if args.failure_category:
        if not args.category_ids_file:
            raise ValueError("--failure_category requires --category_ids_file")
        with open(args.category_ids_file, encoding="utf-8") as f:
            category_ids = json.load(f)
        allowed_ids = set()
        category_value = category_ids.get(args.failure_category)
        if isinstance(category_value, list):
            allowed_ids.update(category_value)
        else:
            for task in args.tasks:
                allowed_ids.update(category_ids.get(task, {}).get(args.failure_category, []))
        if not allowed_ids:
            raise ValueError(
                f"No task ids found for failure_category='{args.failure_category}'. "
                f"Check the structure of {args.category_ids_file}."
            )
        samples = [s for s in samples if s["task_id"] in allowed_ids]
        print(f"Filtered to failure_category='{args.failure_category}': {len(samples)} samples")

    print(f"Loaded {len(samples)} samples")
    random.shuffle(samples)
    all_todos = []

    sharded_extra = f"-at{args.assistant_temperature}-ut{args.user_temperature}" if args.assistant_temperature != 1.0 or args.user_temperature != 1.0 else ""
    st_extra = f"-t{args.assistant_temperature}" if args.assistant_temperature != 1.0 else ""
    ce_prefix = f"{ce_method.name}-" if ce_method is not None else ""
    sharded_ct = f"{ce_prefix}sharded{sharded_extra}"
    full_ct, concat_ct, snowball_ct = f"full{st_extra}", f"concat{st_extra}", f"snowball{sharded_extra}"

    all_tasks = list(set([sample["task"] for sample in samples]))
    for assistant_model in args.models:
        sharded_run_counts, full_run_counts, concat_run_counts, snowball_run_counts = Counter(), Counter(), Counter(), Counter()
        for task in all_tasks:
            sharded_run_counts.update(get_run_counts(sharded_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            full_run_counts.update(get_run_counts(full_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            concat_run_counts.update(get_run_counts(concat_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            snowball_run_counts.update(get_run_counts(snowball_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
        print(f"Sharded run counts: {sharded_run_counts}")
        print(f"Full run counts: {full_run_counts}")
        print(f"Concat run counts: {concat_run_counts}")
        print(f"Snowball run counts: {snowball_run_counts}")

        for sample in samples:
            # all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": full_ct, "system_model": args.system_model, "dataset_fn": dataset_fn}] * (args.N_full_runs - full_run_counts[sample["task_id"]])
            # all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": concat_ct, "system_model": args.system_model, "dataset_fn": dataset_fn}] * (args.N_concat_runs - concat_run_counts[sample["task_id"]])
            # all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": snowball_ct, "system_model": args.system_model, "user_model": args.user_model, "dataset_fn": dataset_fn}] * (args.N_sharded_runs - sharded_run_counts[sample["task_id"]])
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": sharded_ct, "system_model": args.system_model, "user_model": args.user_model, "dataset_fn": dataset_fn}] * (args.N_sharded_runs - sharded_run_counts[sample["task_id"]])

    for todo in all_todos:
        if args.assistant_temperature != 1.0 or args.user_temperature != 1.0:
            todo["assistant_temperature"] = args.assistant_temperature
            todo["user_temperature"] = args.user_temperature
        
        if lu_bandit is not None:
            todo["lu_bandit"] = lu_bandit

        if ce_method is not None:
            todo["ce_method"] = ce_method

    random.shuffle(all_todos)

    print(f"Running {len(all_todos)} conversations")
    print(Counter([todo["assistant_model"] for todo in all_todos]))
    print(Counter([todo["conv_type"] for todo in all_todos]))

    with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
        list(tqdm.tqdm(executor.map(run_simulation, all_todos), total=len(all_todos)))
