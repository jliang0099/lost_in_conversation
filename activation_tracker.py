import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from pathlib import Path


class ActivationTracker:
	def __init__(
		self,
		layers: Optional[list] = None,
		task: Optional[str] = None,
		sample: Optional[str] = None,
		distance_metric: str = "cosine",
		track_full_hidden_states: bool = False,
	):
		self._anchor: Optional[torch.Tensor] = None
		self.layers = layers
		self.task = task
		self.sample = sample
		self.distance_metric = distance_metric
		self.track_full_hidden_states = track_full_hidden_states
		self.activation_history: list[torch.Tensor] = []  # (T, num_layers, D)，用于计算激活距离
		
		# 可选：记录最后一个 token 的 hidden states，shape: (num_layers, D)
		self.final_hidden_states_history: list[dict] = []  # [{"label": "goal"/"turn_i", "hidden_states": tensor(num_layers, D)}]

	def _distances(
		self,
		a: torch.Tensor,
		b: torch.Tensor,
	) -> list[float]:
		"""逐层计算 a 和 b 的距离，shape (L, D) each。"""
		out = []
		for i in range(len(self.layers)):
			ai, bi = a[i], b[i]
			if self.distance_metric == "cosine":
				d = 1.0 - F.cosine_similarity(
					ai.unsqueeze(0), bi.unsqueeze(0)
				).item()
			else:
				d = torch.norm(ai - bi, p=2).item()
			out.append(d)
		return out

	def set_goal(self, hidden_states: list[torch.Tensor]) -> None:
		# prefill 的 hidden_states[layer_idx+1] shape: (1, seq_len, D)
		# 取 -1 即最后一个输入token，语义上是"读完整个goal prompt后的状态"
		hidden = []
		for layer_idx in self.layers:
			h = hidden_states[layer_idx + 1]          # (1, seq_len, D)
			hidden.append(h[0, -1, :].cpu().float())  # (D,)

		self._anchor = torch.stack(hidden)            # (num_layers, D)
		self.activation_history.append(self._anchor)
		
		# 可选：记录最后一个 token 的 hidden states
		if self.track_full_hidden_states:
			self.final_hidden_states_history.append({
				"label": "goal",
				"hidden_states": self._anchor.clone(),  # (num_layers, D)
			})
		# print(f"[GoalDriftTracker] Anchor set | shape={tuple(self._anchor.shape)} | layers={self.layers}")

	def record_activation(self, hidden_states: list[torch.Tensor], turn_id: Optional[int] = None) -> None:
		# 结构完全一样，语义上是"读完当前对话后的状态"
		hidden = []
		for layer_idx in self.layers:
			h = hidden_states[layer_idx + 1]          # (1, seq_len, D)
			hidden.append(h[0, -1, :].cpu().float())  # (D,)

		assert self._anchor is not None, "Anchor not set yet!"
		activation = torch.stack(hidden)
		self.activation_history.append(activation)
		
		# 可选：记录最后一个 token 的 hidden states
		if self.track_full_hidden_states:
			label = f"turn_{turn_id}" if turn_id is not None else f"turn_{len(self.activation_history)-1}"
			self.final_hidden_states_history.append({
				"label": label,
				"hidden_states": activation.clone(),  # (num_layers, D)
			})
		# print(f"[GoalDriftTracker] Activation recorded | layers={self.layers}")

	def generate_result(self):
		assert self._anchor is not None, "Call set_goal() before generate_result()."

		delta_scores: list[float] = []
		delta_per_layer_all: list[list[float]] = []
		cumulative_scores: list[float] = []
		cumulative_per_layer_all: list[list[float]] = []

		# compute delta drift between consecutive activations
		for i in range(1, len(self.activation_history)):
			h_before = self.activation_history[i - 1]
			h_after = self.activation_history[i]
			d_layer = self._distances(h_before, h_after)
			delta_scores.append(float(np.mean(d_layer)))
			delta_per_layer_all.append(d_layer)

		# compute cumulative drift to anchor for each activation
		for h_after in self.activation_history:
			c_layer = self._distances(self._anchor, h_after)
			cumulative_scores.append(float(np.mean(c_layer)))
			cumulative_per_layer_all.append(c_layer)

		return {
			"delta_scores": delta_scores,
			"delta_per_layer": delta_per_layer_all,
			"cumulative_scores": cumulative_scores,
			"cumulative_per_layer": cumulative_per_layer_all,
		}

	def save_hidden_states(self, filepath: str | Path) -> None:
		"""
		将记录的最后一个 token hidden states 保存到文件。
		
		推荐路径格式（与 logs 结构平行）：
		  logs/hidden_states/{task}/{conv_type_config}_{task}_{model}/{conv_id}.pt
		
		支持格式：
		  - .pt / .pth: torch.save() 格式（推荐，保留完整张量和元数据）
		  - .npz: numpy 压缩格式
		"""
		if not self.final_hidden_states_history:
			print("⚠️ No hidden states recorded. Set track_full_hidden_states=True to enable tracking.")
			return
		
		filepath = Path(filepath)
		filepath.parent.mkdir(parents=True, exist_ok=True)
		
		if filepath.suffix in [".pt", ".pth"]:
			# PyTorch 格式：保存完整的张量
			torch.save({
				"metadata": {
					"task": str(self.task),
					"sample": str(self.sample),
					"layers": self.layers,
					"num_records": len(self.final_hidden_states_history),
				},
				"hidden_states": self.final_hidden_states_history,
			}, filepath)
			# print(f"✅ Saved hidden states (PyTorch) -> {filepath}")
		
		elif filepath.suffix == ".npz":
			# NumPy 格式：转换为 numpy，保存为压缩 npz
			save_dict = {
				"task": np.array([self.task], dtype=object),
				"sample": np.array([self.sample], dtype=object),
				"layers": np.array(self.layers),
			}
			for entry in self.final_hidden_states_history:
				label = entry["label"]
				hs = entry["hidden_states"].numpy()  # (num_layers, D)
				save_dict[label] = hs
			
			np.savez_compressed(filepath, **save_dict)
			# print(f"✅ Saved hidden states (NumPy) -> {filepath}")
		
		else:
			raise ValueError(f"Unsupported format: {filepath.suffix}. Use .pt, .pth, or .npz")

	# @staticmethod
	# def aggregate_hidden_states_from_jsonl(jsonl_filepath: str | Path, output_filepath: str | Path) -> dict:
	# 	"""
	# 	从 JSONL 日志文件中聚合所有 conversations 的 hidden states。
		
	# 	用法：
	# 	  ActivationTracker.aggregate_hidden_states_from_jsonl(
	# 	    "logs/math/sharded-at0-ut0_math_meta-llama_Meta-Llama-3-8B-Instruct.jsonl",
	# 	    "logs/hidden_states/math/sharded-at0-ut0_math_meta-llama_Meta-Llama-3-8B-Instruct_aggregated.npz"
	# 	  )
		
	# 	输出 .npz 包含：
	# 	  - 按 turn 聚合的均值张量: goal_mean, turn_0_mean, turn_1_mean, ...
	# 	  - 对应的标准差: goal_std, turn_0_std, ...
	# 	  - 元数据: task, num_conversations, num_turns, layers
	# 	"""
	# 	import json
	# 	from collections import defaultdict
		
	# 	jsonl_path = Path(jsonl_filepath)
	# 	output_path = Path(output_filepath)
	# 	output_path.parent.mkdir(parents=True, exist_ok=True)
		
	# 	# 读取 JSONL，收集所有 hidden states 记录
	# 	records_by_turn = defaultdict(list)  # {turn_label: [(num_layers, D), ...]}
	# 	metadata = {}
	# 	num_valid = 0
		
	# 	with open(jsonl_path, 'r') as f:
	# 		for line in f:
	# 			if not line.strip():
	# 				continue
	# 			try:
	# 				record = json.loads(line)
	# 				# 假设该记录中存有 hidden_states_path 或直接的 hidden_states
	# 				# 这里需要你在日志保存时也记录 hidden_states 路径或数据
	# 				# 简单版本：去掉这个方法，改为在保存后直接聚合
	# 			except:
	# 				pass
		
	# 	if not records_by_turn:
	# 		print("⚠️ No hidden states records found in JSONL")
	# 		return {}
		
	# 	# 计算每个 turn 的汇总统计
	# 	aggregate = {"task": metadata.get("task"), "num_conversations": num_valid}
	# 	for turn_label, states_list in records_by_turn.items():
	# 		if states_list:
	# 			# states_list: [(num_layers, D), ...]
	# 			stacked = np.stack(states_list, axis=0)  # (num_conversations, num_layers, D)
	# 			aggregate[f"{turn_label}_mean"] = np.mean(stacked, axis=0)
	# 			aggregate[f"{turn_label}_std"] = np.std(stacked, axis=0)
		
	# 	np.savez_compressed(output_path, **aggregate)
	# 	print(f"✅ Aggregated hidden states -> {output_path}")
	# 	return aggregate

	# def get_hidden_states_summary(self) -> dict:
	# 	"""获取 hidden states 的元数据摘要。"""
	# 	summary = {
	# 		"track_enabled": self.track_full_hidden_states,
	# 		"num_records": len(self.final_hidden_states_history),
	# 		"layers": self.layers,
	# 	}
	# 	if self.final_hidden_states_history:
	# 		summary["records"] = [
	# 			{
	# 				"label": entry["label"],
	# 				"shape": tuple(entry["hidden_states"].shape),  # (num_layers, D)
	# 				"size_mb": entry["hidden_states"].element_size() * entry["hidden_states"].nelement() / 1e6,
	# 			}
	# 			for entry in self.final_hidden_states_history
	# 		]
	# 	return summary
