from typing import Optional
from activation_tracker import ActivationTracker
from inertia_checker import InertiaChecker
from utils import print_colored, extract_conversation, date_str
from utils_log import log_conversation, save_conversation_hidden_states
from system_agent import SystemAgent
from user_agent import UserAgent
from tasks import get_task
from model_huggingface import generate
from lu_bandit.agent import LUBandit
from ce_methods import CEMethod
from typing import Optional

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
        conv_type: Optional[str] = None,

        lu_bandit: Optional["LUBandit"] = None,
        ce_method: Optional["CEMethod"] = None,
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
        
        # CE-1: For CE methods that modify the system prompt, apply augmentation before the conversation starts.
        if ce_method is not None:
            self.system_message = ce_method.augment_system_message(self.system_message, self.sample)
        self.ce_method: Optional[CEMethod] = ce_method
        self._conv_type_override = conv_type  # full conv_type including CE prefix if any
        
        self.trace = [{"role": "system", "content": self.system_message, "timestamp": date_str()}]
        
        # CE-2a: For CE methods that provide demonstration messages, get them once at the start (e.g. for in-context learning or iterative feedback methods like CoT or Reflexion).
        self._demo_messages = ce_method.get_demo_messages(self.sample) if ce_method is not None else []
        
        # Llama3.1-8B[12, 16, 20, 24, 28] Qwen2.5-14B[10, 20, 30, 40, 46] Qwen3-8B[13, 18, 22, 27, 31] Qwen3-14B[8, 16, 24, 32, 39]
        self.activation_tracker = ActivationTracker(layers=[12, 16, 20, 24, 28], task=self.task, sample=self.sample["task_id"], track_full_hidden_states=True) if track_activation else None

        self.lu_bandit: Optional[LUBandit] = lu_bandit

        # InertiaChecker runs whenever activation tracking is on — independent of LUBandit.
        # goal_drift measures whether CE methods preserve goal alignment across turns.
        self.inertia_checker = InertiaChecker(focus_layer_idx=3) if track_activation else None

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
                    
            generation_messages = extract_conversation(self.trace, to_str=False)
            
            # CE-2b: For CE methods that provide demonstration messages, inject them into the generation messages at the appropriate position (e.g. after the system prompt).
            if self._demo_messages:
                sys_msgs = [m for m in generation_messages if m["role"] == "system"]
                real_msgs = [m for m in generation_messages if m["role"] != "system"]
                generation_messages = sys_msgs + self._demo_messages + real_msgs

            # CE-3: For CE methods that modify the generation messages (e.g. compression methods that shorten conversation history), apply transformation before generation.
            if self.ce_method is not None:
                generation_messages = self.ce_method.transform_generation_messages(
                    generation_messages, n_demo_messages=len(self._demo_messages)
                )

            # LUBandit: select behavioral mode, inject hint into messages
            bandit_meta: dict = {}
            if self.lu_bandit is not None:
                _behavior, generation_messages, bandit_meta = self.lu_bandit.decide(
                    tracker=self.activation_tracker,
                    inertia_checker=self.inertia_checker,
                    messages=generation_messages,
                    n_total_turns=len(shards),
                    task_type=self.task_name,
                )
                if verbose and _behavior != "neutral":
                    print_colored(
                        f"[lu_bandit] behavior={_behavior}  "
                        f"ucb={bandit_meta.get('ucb_score', 0):.3f}",
                        "cyan",
                    )

            print("[debug] messages fed to generate_fn:", generation_messages)
            assistant_response_obj = generate(
                messages=generation_messages,
                model_name=self.assistant_model,
                temperature=self.assistant_temperature,
                max_tokens=max_assistant_tokens,
                is_first_turn=is_first_turn,
                activation_tracker=self.activation_tracker,
                return_metadata=True,
            )

            assistant_response = assistant_response_obj["message"]

            # CE-4: For CE methods that apply post-processing to the assistant response (e.g. self-refinement methods that call generate() again), apply post-processing after generation.
            if self.ce_method is not None:
                assistant_response = self.ce_method.post_process_response(
                    assistant_response=assistant_response,
                    generation_messages=generation_messages,
                    generate_fn=generate,
                    model_name=self.assistant_model,
                    temperature=self.assistant_temperature,
                    max_tokens=max_assistant_tokens,
                )
            
            # CE-5: Record CE token usage in the trace for visibility and logging.
            ce_tokens = self.ce_method.last_turn_ce_tokens if self.ce_method is not None else {}
            
            assistant_trace_entry = {
                "role":      "assistant",
                "content":   assistant_response,
                "timestamp": date_str(),
                "cost_usd":  assistant_response_obj["total_usd"],
                "prompt_tokens":        assistant_response_obj.get("prompt_tokens", 0),
                "completion_tokens":    assistant_response_obj.get("completion_tokens", 0),
                "ce_prompt_tokens":     ce_tokens.get("prompt_tokens", 0),
                "ce_completion_tokens": ce_tokens.get("completion_tokens", 0),
            }
            if bandit_meta:
                assistant_trace_entry["bandit_meta"] = bandit_meta

            #TODO Independently intertia checker
            if self.activation_tracker and self.inertia_checker:
                assistant_trace_entry["inertia"] = self.inertia_checker.summary(
                    self.activation_tracker
                )

            self.trace.append(assistant_trace_entry)
            if verbose:
                print_colored(f"[assistant] {assistant_response}", "red")

            # LUBandit: observe outcome — computes quality, stages reward
            if self.lu_bandit is not None:
                kappa_now = self.lu_bandit.current_kappa(
                    self.activation_tracker, self.inertia_checker
                )
                messages_with_response = extract_conversation(self.trace, to_str=False)
                self.lu_bandit.observe(
                    messages_after=messages_with_response,
                    generate_fn=generate,
                    kappa_after=kappa_now,
                )

            # System verification
            system_verification_response, verification_cost_usd = self.system_agent.verify_system_response(self.trace)
            self.trace.append({"role": "log", "content": {"type": "system-verification", "response": system_verification_response}, "timestamp": date_str(), "cost_usd": verification_cost_usd})
            if verbose:
                print_colored(f"[log] system verification: {system_verification_response}", "blue")

            response_type = system_verification_response["response_type"]

            if response_type == "answer_attempt":
                # Evaluate
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

                # LUBandit: record verification + alignment bonus
                if self.lu_bandit is not None:
                    self.lu_bandit.note_verification(response_type, extracted_answer)

                if is_correct:
                    is_completed = True
                    # LUBandit end-of-episode: flush last reward + task score bonus
                    if self.lu_bandit is not None:
                        kappa_final = self.lu_bandit.current_kappa(
                            self.activation_tracker, self.inertia_checker
                        )
                        self.lu_bandit.end_episode(
                            final_task_score=score,
                            kappa_final=kappa_final,
                        )
                        self.lu_bandit.save_if_configured()
                    self.trace.append({"role": "log", "content": {"type": "conversation-completed", "is_correct": is_correct}, "timestamp": date_str()})
                    if verbose:
                        print_colored(f"[log] conversation completed: {is_correct}; score: {score}", "blue")

            else:
                # Non-answer turn: still record response_type for bandit state + alignment
                if self.lu_bandit is not None:
                    self.lu_bandit.note_verification(response_type)
                if response_type in ["clarification", "discussion"]:
                    continue
            
        # LUBandit end-of-episode for unsolved conversations (flush any staged reward)
        if self.lu_bandit is not None and self.lu_bandit.reward_computer.has_pending():
            kappa_final = self.lu_bandit.current_kappa(
                self.activation_tracker, self.inertia_checker
            )
            self.lu_bandit.end_episode(final_task_score=score, kappa_final=kappa_final)
            self.lu_bandit.save_if_configured()

        assistant_turns = [m for m in self.trace if m["role"] == "assistant"]
        total_prompt_tokens      = sum(m.get("prompt_tokens", 0)        for m in assistant_turns)
        total_completion_tokens  = sum(m.get("completion_tokens", 0)    for m in assistant_turns)
        total_ce_prompt_tokens   = sum(m.get("ce_prompt_tokens", 0)     for m in assistant_turns)
        total_ce_completion_tokens = sum(m.get("ce_completion_tokens", 0) for m in assistant_turns)
        self.trace.append({
            "role": "log",
            "content": {
                "type": "token-usage",
                "prompt_tokens":            total_prompt_tokens,
                "completion_tokens":        total_completion_tokens,
                "total_tokens":             total_prompt_tokens + total_completion_tokens,
                "ce_prompt_tokens":         total_ce_prompt_tokens,
                "ce_completion_tokens":     total_ce_completion_tokens,
                "ce_total_tokens":          total_ce_prompt_tokens + total_ce_completion_tokens,
                "grand_total_tokens":       (total_prompt_tokens + total_completion_tokens
                                             + total_ce_prompt_tokens + total_ce_completion_tokens),
            },
            "timestamp": date_str(),
        })
        if verbose:
            print_colored(
                f"[log] token usage — main: {total_prompt_tokens}p/{total_completion_tokens}c"
                + (f", CE overhead: {total_ce_prompt_tokens}p/{total_ce_completion_tokens}c"
                   if total_ce_prompt_tokens or total_ce_completion_tokens else ""),
                "blue",
            )

        if save_log:
            if self._conv_type_override is not None:
                conv_type = self._conv_type_override
            elif self.run_with_custom_temperature:
                conv_type = f"sharded-at{self.assistant_temperature}-ut{self.user_temperature}"
            else:
                conv_type = "sharded"
            conv_id = log_conversation(
                conv_type, self.task.get_task_name(), self.sample["task_id"],
                self.dataset_fn, self.assistant_model, self.system_model,
                self.user_model, self.trace, is_correct, score,
                log_folder=self.log_folder,
            )
            # Activation tracker log
            if self.activation_tracker:
                save_conversation_hidden_states(
                    conv_id, conv_type, self.task.get_task_name(),
                    self.assistant_model, self.activation_tracker,
                    log_folder=self.log_folder
                )
        return is_correct, score
