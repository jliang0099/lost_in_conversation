import json
from utils import extract_conversation
from model_huggingface import generate_json
from tasks import get_task


class SystemAgent:
    def __init__(self, task_name, system_model, sample):
        self.system_model = system_model
        self.task_name = task_name
        self.task = get_task(task_name)
        self.answer_extraction_strategy = self.task.answer_extraction_strategy
        self.sample = sample
        self.answer_description = self.task.get_answer_description()
        self.max_extraction_attempts = 3

        assert self.answer_extraction_strategy in ["full_response", "prefix_suffix", "gen", "task_specific"], f"Answer extraction strategy {self.answer_extraction_strategy} not supported"

        with open("prompts/system_turn_categorization.txt", "r", encoding="utf-8") as f:
            self.system_verification_prompt = f.read()
        with open("prompts/system_answer_extraction_gen.txt", "r", encoding="utf-8") as f:
            self.answer_extraction_prompt_gen = f.read()
        with open("prompts/system_answer_extraction_prefix_suffix.txt", "r", encoding="utf-8") as f:
            self.answer_extraction_prompt_prefix_suffix = f.read()


    def verify_system_response(self, conversation_so_far):
        if self.task_name in ["summary", "totto", "translation"]:
            # in these tasks, the assistant is explicitly instructed to provide an answer attempt at each turn
            return {"response_type": "answer_attempt"}, 0.0

        initial_query = self.sample["shards"][0]["shard"]
        shards = self.sample["shards"][1:]

        last_turn_text = extract_conversation(conversation_so_far, to_str=True, only_last_turn=True)

        # print("--------------------- TURN CLASSIFICATION ---------------------")
        # print(f"[debug] last_turn_text:\n{last_turn_text}\n---")

        system_verification_prompt_populated = self.system_verification_prompt.replace("[[CONVERSATION_SO_FAR]]", last_turn_text).replace("[[INITIAL_SHARD]]", initial_query).replace("[[SHARDS]]", json.dumps(shards)).replace("[[ANSWER_DESCRIPTION]]", self.answer_description)
        system_verification_response_obj = generate_json([{"role": "user", "content": system_verification_prompt_populated}], model=self.system_model, return_metadata=True, temperature=0.0)
        system_verification_response = system_verification_response_obj["message"]

        # print(f"[debug] system_verification_response:\n{system_verification_response}\n---")
        # print("--------------------- END TURN CLASSIFICATION ---------------------")

        return system_verification_response, system_verification_response_obj["total_usd"]

    def extract_answer(self, conversation_so_far):
        assistant_response = [msg["content"] for msg in conversation_so_far if msg["role"] == "assistant"][-1]
        
        if self.answer_extraction_strategy == "full_response":
            return assistant_response # just return the full response
        elif self.answer_extraction_strategy == "task_specific":
            return self.task.extract_answer(assistant_response)
        else:
            prompt = self.answer_extraction_prompt_gen if self.answer_extraction_strategy == "gen" else self.answer_extraction_prompt_prefix_suffix
            last_assistant_turn_text = extract_conversation(conversation_so_far, to_str=True, only_last_turn=True)
            extracted_answer = None
            extraction_attempts = 0
            while extracted_answer is None and extraction_attempts < self.max_extraction_attempts:
                extraction_attempts += 1
                answer_extraction_response_obj = generate_json([{"role": "user", "content": prompt}], model=self.system_model, return_metadata=True, variables={"ASSISTANT_RESPONSE": last_assistant_turn_text, "ANSWER_DESCRIPTION": self.answer_description}, temperature=0.0)
                answer_extraction_response = answer_extraction_response_obj["message"]
                if self.answer_extraction_strategy == "gen":
                    extracted_answer = answer_extraction_response["answer"]

                else:
                    extractor_response = answer_extraction_response["answer"]
                    if "[...]" in extractor_response and extractor_response.count("[...]") == 1 :
                        prefix, suffix = extractor_response.split("[...]")
                        prefix, suffix = prefix.strip(), suffix.strip()

                        start_idx = assistant_response.find(prefix)
                        end_idx = assistant_response.rfind(suffix)
                        extracted_answer = assistant_response[start_idx:(end_idx+len(suffix))]
                    else:
                        extracted_answer = extractor_response

                if extracted_answer is not None and extracted_answer not in assistant_response:
                    extracted_answer = None # will need to try again, this ensures the process is extractive

        if extracted_answer is None:
            print(f"Failed to extract answer after {extraction_attempts} attempts")
            extracted_answer = "" # defaulting to empty string
        return extracted_answer
