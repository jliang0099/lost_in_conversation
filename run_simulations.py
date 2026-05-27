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
from steering.steering import SteeringController
from context_compressor import ContextCompressor
from mediator import MediatorAgent

def run_simulation(todo):
    dataset_fn = todo["dataset_fn"]
    try:
        assistant_temp = todo.get("assistant_temperature", 0)
        user_temp = todo.get("user_temperature", 0)
        # steering_controller = todo.get("steering_controller")

        if todo["conv_type"].startswith("sharded"):
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

                inertia_check=todo.get("inertia_check", False),
                inertia_curvature_threshold=todo.get("inertia_curvature_threshold", -0.23),
                # inertia_var_slope_threshold=todo.get("inertia_var_slope_threshold", 0.0),

                context_compressor=todo.get("context_compressor", None),
                mediator=todo.get("mediator", None),
                # steering_controller=steering_controller
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

    parser.add_argument("--inertia_check", action="store_true", default=False,
                        help="Enable inertia-based prompt intervention (curvature + variance checks)")
    parser.add_argument("--inertia_curvature_threshold", type=float, default=-0.23,
                        help="Temporal curvature κ below this triggers intervention (default: -0.2)")

    parser.add_argument("--context_compress", action="store_true", default=False,
                        help="Enable LLMLingua-2 context compression before generation")
    parser.add_argument("--compress_rate", type=float, default=0.5,
                        help="Target compression rate for LLMLingua-2 (default: 0.5)")
    parser.add_argument("--compress_keep_last_n_turns", type=int, default=1,
                        help="Number of recent turns to keep verbatim (default: 1)")

    parser.add_argument("--mediator", action="store_true", default=False,
                        help="Enable Mediator-Assistant framework (requires experiences/{task}.json)")
    parser.add_argument("--mediator_model", type=str, default=None,
                        help="Model for the Mediator LLM call (defaults to --system_model)")
    
    # parser.add_argument("--inertia_var_slope_threshold", type=float, default=0.0,
    #                     help="Variance slope below this triggers intervention (default: 0.0)")

    # parser.add_argument("--steering_artifacts_dir", type=str, default=None,
    #                     help="Directory containing layer_L.pt steering artifacts. "
    #                          "If omitted, no steering is applied.")
    # parser.add_argument("--steer_alpha", type=float, default=0.5,
    #                     help="Base proportional steering strength (used when --steering_artifacts_dir is set).")

    args = parser.parse_args()

    # Build ContextCompressor once; shared (read-only) across all worker threads.
    context_compressor = None
    if args.context_compress:
        context_compressor = ContextCompressor(
            rate=args.compress_rate,
            keep_last_n_turns=args.compress_keep_last_n_turns,
        )
        print(f"[context_compress] rate={args.compress_rate}  keep_last_n_turns={args.compress_keep_last_n_turns}")

    # Build MediatorAgent once per task; shared across worker threads.
    # MediatorAgent is stateless after init, so sharing is safe.
    mediator_map = {}  # task -> MediatorAgent
    if args.mediator:
        mediator_model = args.mediator_model or args.system_model
        for task in args.tasks:
            try:
                mediator_map[task] = MediatorAgent(
                    task=task,
                    model=mediator_model,
                    experiences_dir="experiences",
                )
                print(f"[mediator] loaded experiences for task='{task}' using model='{mediator_model}'")
            except FileNotFoundError as e:
                print(f"[mediator] WARNING: {e} — mediator disabled for task='{task}'")

    # Build SteeringController once; shared (read-only) across all worker threads.
    # if args.steering_artifacts_dir is not None:
    #     steering_controller = SteeringController(args.steering_artifacts_dir, base_alpha=args.steer_alpha)
    #     print(f"[steering] artifacts_dir={args.steering_artifacts_dir}  base_alpha={args.steer_alpha}")
    # else:
    #     steering_controller = None

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

    # samples = [sample for sample in samples if len(sample["shards"]) >= 6]

    print(f"Loaded {len(samples)} samples")
    random.shuffle(samples)
    all_todos = []

    sharded_extra = f"-at{args.assistant_temperature}-ut{args.user_temperature}" if args.assistant_temperature != 1.0 or args.user_temperature != 1.0 else ""
    st_extra = f"-t{args.assistant_temperature}" if args.assistant_temperature != 1.0 else ""
    sharded_ct, full_ct, concat_ct, snowball_ct = f"sharded{sharded_extra}", f"full{st_extra}", f"concat{st_extra}", f"snowball{sharded_extra}"

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
        
        if args.inertia_check:
            todo["inertia_check"] = True
            todo["inertia_curvature_threshold"] = args.inertia_curvature_threshold

        if context_compressor is not None:
            todo["context_compressor"] = context_compressor

        if mediator_map:
            task = todo["sample"]["task"]
            if task in mediator_map:
                todo["mediator"] = mediator_map[task]
        #     todo["inertia_var_slope_threshold"] = args.inertia_var_slope_threshold
        
        # todo["steering_controller"] = steering_controller  # None = no steering

    random.shuffle(all_todos)

    print(f"Running {len(all_todos)} conversations")
    print(Counter([todo["assistant_model"] for todo in all_todos]))
    print(Counter([todo["conv_type"] for todo in all_todos]))

    with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
        list(tqdm.tqdm(executor.map(run_simulation, all_todos), total=len(all_todos)))
