---
name: project-ce-methods
description: CE方法架构设计、三个已实现方法（FewShot/CoT/SelfRefine）的完整细节、接入点、conv_type命名、扩展规则
metadata:
  type: project
---

## 项目背景

lost_in_conversation：研究 sharded（分片）数学对话任务中不同 Context Engineering 方法对 LLM 的影响。GSM8K 问题拆成多个 shards，user agent 逐轮透露信息，assistant 需累积信息后作答。评测对象目前主要是 `meta-llama/Llama-3.1-8B-Instruct`（vLLM on port 5001）。

**Why:** 论文级实验，多个 CE 方法需干净可分，各方法独立 log，互不污染，方便单独评估各方法的 accuracy 提升。

---

## 架构：ce_methods/ 包

```
ce_methods/
  __init__.py      # build_ce_method(name, **kwargs) 工厂 + CHOICES 注册表
  base.py          # CEMethod 基类，定义两个钩子
  few_shot.py      # FewShotCE
  cot.py           # CoTCE
  self_refine.py   # SelfRefineCE
```

### 两个钩子（base.py）

| 钩子 | 签名 | 调用时机 | 默认行为 |
|------|------|---------|---------|
| `augment_system_message` | `(system_msg: str, sample: dict) -> str` | simulator `__init__` 时，对话开始前 | 原样返回 |
| `post_process_response` | `(response, gen_msgs, generate_fn, model_name, temperature, max_tokens) -> str` | 每轮 assistant 生成后、trace 写入前（步骤 3a） | 原样返回 |

### 接入点

**huggingface_simulator_sharded.py**
- `__init__`（L65-67）：`ce_method.augment_system_message()` 修改 `self.system_message`，trace[0] 用改后的 system message
- `run()` 步骤 3a（L180前后）：生成 `assistant_response` 后立即调用 `ce_method.post_process_response()`，返回值替换原 response 写入 trace

**run_simulations.py**
- `build_ce_method()` 在 `args = parser.parse_args()` 后立即构建，共享给所有 worker（stateless 方法线程安全）
- conv_type 前缀逻辑：
  ```python
  ce_prefix = f"{ce_method.name}-" if ce_method is not None else ""
  sharded_ct = f"{ce_prefix}sharded{sharded_extra}"
  ```
- todo 里加 `"ce_method": ce_method`，`run_simulation` 用 `"sharded" in conv_type` 判断路由

### CLI 参数

```bash
--ce_method     choices: none/fewshot/cot/self_refine  (default: none)
--fewshot_k     int, default=3
--fewshot_log   demo log 路径，default: logs/math/sharded-at0-ut0/(260-428)...jsonl
```

---

## 三个方法详解

### 1. FewShotCE（ce_methods/few_shot.py）

**原理**：给 demo（imitation learning），不给推理指令。
**钩子**：仅 `augment_system_message`。

**Demo 来源与无污染保证**
- 文件：`logs/math/sharded-at0-ut0/(260-428)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl`
- 评测集用 task_id 40-103，demo 用 260-428，完全不重叠
- 当前 task_id 如果恰好出现在 demos 里会被自动剔除（`augment_system_message` 里过滤）

**Demo 选取策略**
- 过滤：`is_correct=True` + `user_turns <= max_demo_turns`（default 4，排除 7-turn 怪物）
- 分层抽样：按 user-turn 数排序后均匀取 k 个，保证 short/medium/longer 多样性
- 默认 k=3，选出的 demo 示例（turns=1, 4, 4）

**demo_style="clean"**（核心设计决策）
- 问题：log 里存的是 Llama 原始行为，含 hedging（"I'll provide general scenarios for different cases..."），不适合作为 ideal demo
- 解决：intermediate turns 的 assistant response 全部替换为一行 echo 模板：
  `Noted — "{user_msg[:100]}". I'll incorporate this once I have all the information.`
- final turn 保留原始计算 + 答案，用 sentence-boundary 截断（`_truncate_at_sentence`）：
  优先 `".\n"` 边界（bullet-point 响应），再退回 `". [A-Z]"`（散文句），最后 word boundary
- `max_final_chars=1000`（Llama 响应多在 500-1000 chars，完整保留避免截断到计算中途）

**token 开销**：~880 tokens（k=3，仅第一轮 system message 多出来这些）

---

### 2. CoTCE（ce_methods/cot.py）

**原理**：零样本推理过程指令（zero-shot CoT），不给例子，告诉模型 HOW to think。
**钩子**：仅 `augment_system_message`。

**与 FewShot 的区别**
- FewShot 给 WHAT（通过示例让模型模仿）
- CoT 给 HOW（通过协议让模型自主推理）
- system message 追加的内容完全不同，语义上互斥，不应组合使用

**注入内容（`_PROTOCOL`）**
```
两阶段协议：
Phase 1 — Accumulation（还没有足够信息时每轮执行）：
  • 列出已收集的数值事实
  • 说明还在等什么信息
  • 禁止猜测或给出 partial answer

Phase 2 — Solution（拿到全部信息的那一轮）：
  • 一行列出所有已知量
  • 逐步写出每个算术操作：运算、值、结果
  • 最后一行给出单个数字答案
```

**token 开销**：~170 tokens（纯指令，无例子）

---

### 3. SelfRefineCE（ce_methods/self_refine.py）

**原理**：Madaan et al. (2023) Self-Refine，生成后自我 critique + 精修。
**钩子**：仅 `post_process_response`（完全不碰 system message）。

**与 FewShot/CoT 的区别**
- FewShot/CoT 是 prompt-level 干预（修改 system message）
- SelfRefine 是 generation-level 干预（在生成后再调用一次 generate）
- 三者钩子完全不重叠，可以理论上叠加（但目前不建议）

**机制**
每轮 assistant 生成 `response` 后：
1. 构建 critique_messages = generation_messages + `[{assistant: response}, {user: CRITIQUE_PROMPT}]`
2. 调用 `generate_fn(critique_messages, ...)` 得到精修版本
3. 精修版本替换原 response 写入 trace

**`_CRITIQUE_PROMPT`** 覆盖两种情况：
- 还在 accumulation 阶段 → 确认已知事实，保持简洁
- 给出了答案 → 逐步验证算术，改正错误，最后重述单个数字答案

**额外开销**：每轮 +1 次 `generate_fn` 调用（4 轮对话 = 原来 2 倍 compute）。不传 `activation_tracker`，不干扰 hidden state 记录。

---

## 方法对比一览

| | FewShot | CoT | SelfRefine |
|---|---|---|---|
| 钩子 | `augment_system_message` | `augment_system_message` | `post_process_response` |
| 修改 system message | ✅ | ✅ | ❌ |
| 额外 generate 调用 | ❌ | ❌ | ✅ 每轮 +1 |
| token 开销/turn | ~880（仅首轮） | ~170（仅首轮） | +整个 context |
| conv_type 前缀 | `fewshot-` | `cot-` | `self_refine-` |
| 日志路径示例 | `logs/math/fewshot-sharded-at0-ut0/` | `logs/math/cot-sharded-at0-ut0/` | `logs/math/self_refine-sharded-at0-ut0/` |

---

## 扩展规则（保持 CE 方法可分）

1. **单一职责**：每个方法只覆盖它应该覆盖的钩子；system-message 方法不做额外 generate，generation 方法不改 system message
2. **注册**：`name` 唯一，加入 `ce_methods/__init__.py` 的 `_REGISTRY`
3. **语义隔离**：FewShot 给例子，CoT 给推理流程，不要在单个方法里同时做两件事
4. **CLI 参数**：新增方法的超参以 `--{name}_xxx` 格式命名，`build_ce_method` 工厂里加对应 kwargs 传递

**How to apply:** 每次添加新 CE 方法时检查：① 只覆盖了该覆盖的钩子；② name 已注册；③ run_simulations.py 的 build_ce_method 调用传了正确的 kwargs。
