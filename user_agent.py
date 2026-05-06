import json

from model_huggingface import generate_json
from utils import extract_conversation

class UserAgent:
    def __init__(self, task, model="gpt-4o"):
        # self.model = model
        self.task = task
        self.task_name = task.get_task_name()
        # with open("prompts/user_agent.txt", "r") as f:
        #     self.prompt_response = f.read()

    def generate_response(self, conversation, sample, temperature=1.0):
        num_user_msgs = sum(1 for msg in conversation if msg["role"] == "user")

        if self.task_name in ["translation", "summary", "data2text"]:
            return self.task.populate_sharded_prompt(sample, num_user_msgs)

        if num_user_msgs == 0:
            shard_id = -1
            first_shard = sample["shards"][0]
            shard_id = first_shard["shard_id"]
            initial_query = first_shard["shard"]

            return initial_query, shard_id, 0.0

        else: # Turn 2 ... n            
            shard_revealed_ids = [msg["content"]["shard_id"] for msg in conversation if msg["role"] == "log" and msg["content"]["type"] == "shard_revealed"]

            shards = sample["shards"][1:] # filter out the first shard, which is the initial query
            shard_ids = [shard["shard_id"] for shard in shards]
            
            # shard_ids_revealed = [shard_id for shard_id in shard_ids if shard_id in shard_revealed_ids]
            shard_ids_not_revealed = [shard_id for shard_id in shard_ids if shard_id not in shard_revealed_ids]

            # shard_texts_revealed = [shard for shard in shards if shard["shard_id"] in shard_ids_revealed]
            shard_texts_not_revealed = [shard for shard in shards if shard["shard_id"] in shard_ids_not_revealed]
            
            # shards_revealed_str = json.dumps(shard_texts_revealed)
            # shards_not_revealed_str = json.dumps(shard_texts_not_revealed)

            # user_agent_prompt_populated = self.prompt_response.replace("[[CONVERSATION_SO_FAR]]", extract_conversation(conversation, to_str=True, skip_system=True)).replace("[[SHARDS_REVEALED]]", shards_revealed_str).replace("[[SHARDS_NOT_REVEALED]]", shards_not_revealed_str)
            # response_obj = generate_json([{"role": "user", "content": user_agent_prompt_populated}], model=self.model, timeout=100, return_metadata=True, temperature=temperature)
            # response = response_obj["message"]
            
            if not shard_ids_not_revealed:
                return "I don't know", -1, 0.0

            # return response["response"], response["shard_id"], response_obj["total_usd"]
            
            next_shard = shard_texts_not_revealed[0]
            return next_shard["shard"], next_shard["shard_id"], 0.0
