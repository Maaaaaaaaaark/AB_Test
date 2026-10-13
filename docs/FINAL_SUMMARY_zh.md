# 项目最终总结（中文）

worker 模型评测项目已完成实现与本地验证，**从未在本地运行任何真实推理**。
本机仅做开发与测试；所有 GPU 推理都在集群上执行。

---

## 1. 实现了什么

一个干净、独立的评测项目（`D:\AB_Text`，无训练、无 MAS 依赖），评估 5 个冻结
模型配置在两个 worker 角色上的能力：

**5 个配置**（模型/tokenizer/chat template 均按 SHA 固定版本；绝不使用相似名替代）：

| 配置名 | 模型 | 思考模式 |
|---|---|---|
| qwen3-4b-thinking-off | Qwen/Qwen3-4B | thinking 关闭（预填空 `<think>` 块）|
| qwen3-4b-thinking-on | Qwen/Qwen3-4B | thinking 开启 |
| gemma-3-4b-it | google/gemma-3-4b-it | 无思考通道（gated）|
| llama-3.2-3b-instruct | meta-llama/Llama-3.2-3B-Instruct | 无思考通道（gated）|
| deepseek-r1-distill-qwen-1.5b | DeepSeek-R1-Distill-Qwen-1.5B（官方仓库名）| 预填式思考（`<｜Assistant｜><think>` 预填）|

**两个 worker 角色与任务：**

1. **证据型本地 QA + 弃权（evidence worker）**
   - MuSiQue-Ans 本地任务：每个父问题选一个子问题（带种子选择，能避开首跳就避开）；
     `#N` 引用用先决子问题答案确定性内联替换（从不交给 LLM 改写）；5 段可见文本
     = 标注支撑段 + 4 段"对该父问题任何子问题都不构成支撑"的段落（严格规则，
     池不足则剔除该父问题并计数）；永不泄露当前/后续答案与支撑标签；保留
     `X >> relation` 语法并在指令中统一解释；gold = 规范子答案（仅当规范子答案与
     父答案在官方归一化下相等时，末步才并入父答案别名，并记录别名来源）。
   - 配对负样本：支撑段被同位置的第 5 段非支撑段替换，指令完全相同，期望输出
     恰好 `NOT FOUND`；构建时筛查（答案/别名出现在标题/段落/先决答案中）、
     是/否与纯数字负样本按政策排除；全部进入人工审核文件（绝不自动批准），
     只有 approved + human_reviewed 的配对进入正式弃权评估。
   - 评分：官方 AnswerMetric（EM/F1），另计弃权相关指标（错误弃权、负样本正确弃权）。
2. **数学推理检查（math checker）**：ProcessBench——官方数据（4 个子集
   400/1000/1000/1000 条，固定 revision）、官方批评提示模板、官方抽取
   （全文最后一个 `\boxed{}` → 整数，含 -1）、官方评分（找首个错误步骤，
   error acc / correct acc / F1=2ab/(a+b)）；4 个子集分数分别报告 + 项目宏平均。
   不加干扰项。
3. **整题参照任务**（描述性参照，不参与选择）：HotpotQA distractor
   （全题 + 10 段，官方 normalize/f1）、GSM8K 主测试（官方 `####` 抽取器，
   官方正则为主、放宽解析仅作诊断、报告抽取失败率）、原始 MATH
   （`\boxed{}` 抽取与等价判定分开；MATH-500 禁用；7500/5000 条数与字段已验证；
   原始 URL 403 → 镜像经记录待批准，见 UNRESOLVED #2）。

**数据与划分（全部可复现、来源记录在 provenance 文件里）：**
- MuSiQue：train → debug 池（smoke/pilot 开发用）；dev → selection 100 父问题
  + holdout 100 父问题（互斥）。
- ProcessBench：项目自定义划分，按整题组（同一题目的全部步骤变体一起走），
  组顺序 = 官方首现顺序 + `order_seed` 确定性洗牌；selection/holdout 各
  25/子集（整组取满，实际 25–26/子集、共 101 条）、pilot 从 debug 组整组选取、
  错误率比例逐划分报告。
- 跨数据集查重：归一化精确匹配 + 近重复（Jaccard≥0.95 标记、仅提示不自动排除）；
  归一化保留数字与运算符、运算符相邻空白视为无意义（"x + 2 = 5" = "x+2=5"），
  但绝不合并字母数字 token（"x 2" ≠ "x2"）。参照子集冻结（每数据集 100 条、
  seed 0），排除本地 debug/selection 用过的题并报告。查重 ≠ 无污染声明。
- 人工审核文件：`artifacts/manifests/musique_negative_review.json`，210 条
  （176 proposed + 34 构建期排除）。**人工决定在重建后保留**（human_reviewed
  条目的决定字段跨 rebuild 保留），人工排除的配对在构建覆盖率里记为
  `review:<reason>`。

**生成协议（所有配置一致；每条记录保存与模型默认值合并后的完整参数）：**
- BF16、无量化、eval 模式、transformers 后端、batch 1、do_sample=True、
  temperature=0.6、top_p=0.95、top_k=0、repetition_penalty=1.0。
- 每 item 每 seed 生成一次；逐项种子由 (run seed, item id) 稳定派生、与顺序无关。
- 正式本地 run 用 seed 0/1/2（每个 seed 单独报告 + 均值；绝不做 best-of-N /
  多数投票）；smoke/pilot/参照用 seed 0。
- Token 预算（安全预算）：MuSiQue 2048 新 token；ProcessBench/整题任务 4096
  （各配置一致；8192 属于另一种预算条件，未启用）。
- 生成前逐项做上下文长度检查（统一资格规则），放不下的记录为不适用，绝不静默截断。
- 原始 token ID + 原始文本在任何清洗之前落盘。
- 提示渲染使用 vendored、sha256 锁定的原生 chat template，与 transformers 的
  jinja 环境完全一致（trim/lstrip、StrictUndefined、tojson），并像 transformers
  一样传 `tools=None`/`documents=None`（Qwen3 模板的 `{% if tools %}` 分支依赖它）。

**解析与计分规则：**
- 按模型测试过的思考区解析器（prefilled/optional/disabled/no_channel 四种模式）；
  从不从思考区内取答案；生成结束时思考区未闭合 = 缺最终答案（记 missing_final）；
  DeepSeek 预填场景（只有闭合 `</think>`）显式处理。
- 达到 token 上限 = 正常评估结果（按截断文本计分）；超时/OOM/崩溃 = 基础设施
  失败（不部分计分，同一 item+seed 最多按 max_attempts_per_item 重试，
  绝不为错误答案重试），未解决则对比记为不完整。
- 计分器全部 vendor 自官方实现（见下文清单），差异逐条记录。

**指标与统计：**
- Tin = 应用原生模板后的输入 token（含预填前缀）；Tout = 全部生成 token
  （含思考 + EOS）。
- Cproxy = 2 × N_text × (Tin+Tout)；N_text 只读 safetensors 头 + config.json：
  `tie_word_embeddings=true` 且嵌入/LM head 形状相等时这对权重只计一次
  （transformers get_num_parameters 约定），Gemma 视觉塔与
  multi_modal_projector 排除，明细审计可查（`weval inspect-params`）。
- Cweighted 无来源价格时 = null（绝不为 0、绝不编造）。
- Efficiency_compute = mean(score∈[0,1]) / mean(Cproxy/1e12)，仅同一任务内计算。
- 正/负样本成本分开报告；50:50 正负混合是成本比较的**声明假设**，不是部署占比估计。
- 统计：逐 seed + 跨 seed 均值、配对配置差异、按父问题组聚类的 95% bootstrap CI
  （配对/变体/seed 一起重采样）、宏平均保持子集结构。

**选择规则（configs/thresholds.yaml，正式选择前需用户确认）：**
- evidence worker：可回答 F1 ≥ 0.80、错误弃权 ≤ 0.10、人工审核负样本正确弃权 ≥ 0.90。
- math checker：ProcessBench 宏 F1 ≥ 0.60 且宏 correct-solution acc ≥ 0.50 且
  宏 error-solution acc ≥ 0.50。
- 合格者中按角色工作负载（50/50 声明混合）最低平均 Cproxy；报告质量-成本帕累托；
  不强制选赢家；holdout 只跑 selection 报告提名的候选（runner 强制执行，禁止
  用 holdout 重新挑选）；阈值文件当前 `draft_unconfirmed`（用户确认即解锁）。

**工程性质：**
- 模块化（数据适配器 / manifest 构建 / 提示渲染 / 后端接口 vLLM-ready 但先
  transformers / 思考解析 / 官方计分包装 / 指标与报告）；逐项增量落盘 + 精确
  resume（缓存键含模型 revision、模板、提示、item、生成参数、seed、解析/计分版本）。
- `docs/worked_examples.md`：真实输入渲染示例（gold 单独存放；未运行过的生成
  明确标 pending，绝不编造输出）；英文 `README.md`（精确安装与执行步骤）；
  可配置 Slurm 脚本（账户等通过 env.sh 覆盖，无硬编码）；pilot 成本/时间外推
  （带不确定性、明确标注为估计）；`docs/UNRESOLVED.md` 记录数据/权限/审核缺口。
- 测试：**245 个全部通过**（覆盖泄漏、整组划分、别名、缺答案计分、思考解析、
  超时、截断、成本计数、resume、模板锁定、种子、渲染、审核往返等）。

**本轮修复的六个模块问题**（子代理审计发现，已修复并有回归测试锁定）：
1. 模板渲染缺 `tools=None`（Qwen3 模板在 StrictUndefined 下会崩 → 集群阻塞级）——已修。
2. `N_text` 平铺去重按 key 名而非参数去重，tied 权重被重复计数——已改为形状判等。
3. 归一化未消除运算符旁空白，削弱跨数据集泄漏检测——已修（且不制造 token 边界）。
4. `partition_records` 接受 order_seed 但从不使用——已改为确定性洗牌（同种子可复现）。
5. `_select_items` 会切开 ProcessBench 整组——pilot/selection/holdout 改为整组选取
   （smoke 保留扁平上限，属开发屏幕的明文例外）。
6. 重建会重写审核文件、抹掉人工决定——已实现 human_reviewed 条目的决定字段跨
   rebuild 保留，人工排除的配对计入构建报告。

---

## 2. 官方 vs 自定义清单

| 任务 | 官方（逐字 vendor） | 项目自定义（已注明） |
|---|---|---|
| MuSiQue | AnswerMetric（normalize_answer、EM/F1） | 本地任务设计：单子问题、`#N` 内联、5 段可见文本、弃权协议、配对负样本、人工审核流程 |
| ProcessBench | 数据（固定 revision）、批评提示模板、最后 `\boxed{}` 抽取、找首个错误步骤、子集 F1 | 划分（整题组 + 种子洗牌）、宏平均（F1 与两类 acc）、schema 说明（pinned 版本无 final_answer 列） |
| GSM8K | 主测试集、`####` 官方正则抽取（首个匹配）、INVALID_ANS 规则 | 官方解析失败时给出放宽解析作为**诊断**并报告失败率；逗号剥离行为由测试锁定 |
| MATH | `\boxed{}` 抽取、官方 math_equivalence.py（等价判定不做全符号求解声明） | 7500/5000 与字段验证；原始 URL 403 → EleutherAI 镜像（固定 revision，**待批准**）；MATH-500 禁用 |
| HotpotQA | normalize_answer / f1 | 只评答案（不评证据链）；10 段干扰项为数据集自带设置 |

## 3. 集群命令

```bash
# 1) 把项目目录（含 artifacts/data、artifacts/manifests，约 1 GB）传到集群
# 2) 配置环境：cp scripts/slurm/env.example.sh scripts/slurm/env.sh
#    填写 SBATCH_ACCOUNT / PARTITION / GPUS / CPUS / MEM / TIME / LOG_DIR / HF_HOME
#    （HF_TOKEN 仅在集群侧设置，永不提交到仓库）

bash scripts/slurm/submit.sh precheck        # 硬件/许可检查，锁定 gated 模板 hash
bash scripts/slurm/submit.sh snapshots       # 按固定 revision 下载 5 个模型快照
bash scripts/slurm/submit.sh inspect-params  # 生成 N_text（5 配置）
bash scripts/slurm/submit.sh smoke           # smoke：backend=transformers，seed 0
REPORT_PLAN=smoke bash scripts/slurm/submit.sh report

# 人工审核负样本（本地/集群均可）：
weval review list --only proposed
weval review approve --parent <id>
weval review exclude --parent <id> --note "<原因>"
# 审核后重建一次，让批准结果进入 manifest：
weval build

bash scripts/slurm/submit.sh pilot           # pilot（要求有已批准配对）
REPORT_PLAN=pilot bash scripts/slurm/submit.sh report
weval estimate --source pilot --target selection
weval estimate --source pilot --target holdout
# 审阅 pilot 报告后：确认 configs/thresholds.yaml（status: confirmed）
bash scripts/slurm/submit.sh selection       # ← 仅在用户明确批准后启动
REPORT_PLAN=selection bash scripts/slurm/submit.sh report
bash scripts/slurm/submit.sh holdout         # ← 仅在用户明确批准后启动（只跑提名配置）
bash scripts/slurm/submit.sh reference       # 整题参照（不参与选择）
```

## 4. 成本估计（当前状态，诚实说明）

- 当前 `artifacts/estimates/*` 来自 **scripted smoke 记录**，只证明外推管线可用；
  wall time 无意义、Cproxy 因尚无模型快照（N_text 缺失）为 null。
- 真实成本/时间估计的流程：集群 pilot（真实推理）→
  `weval estimate --source pilot --target selection/holdout` →
  按配置×任务给出总 item 数、墙钟小时（含组聚类 95% CI）、总 token、
  Cproxy；Cweighted 在提供来源价格前保持 null。
- 规模参考：pilot = 每配置 musique 10 父问题（正 10 + 已批准负 ≈ 8，共约 18 条）
  + processbench 5/子集整组取满（21 条）+ 整题参照各 10 条 × 3 数据集，seed 0；
  selection/holdout = 每配置 musique 100 父（正 100 + 负约 88/80 条）+
  processbench 101 条 + 参照 100×3，各 seed 0/1/2。

## 5. 剩余待批准事项（绝不自动启动）

1. **模型访问**（UNRESOLVED #1）：在集群 HF 账号上接受 Gemma 与 Llama 许可，
   集群侧配置 HF_TOKEN，然后 `weval precheck` 锁定这两个 gated 模板。
2. **MATH 镜像签署**（UNRESOLVED #2，PENDING-APPROVAL）：原始 URL 403，现用
   EleutherAI/hendrycks_math（固定 revision、7500/5000 已验证）——需用户一句确认。
3. **人工审核 176 个负样本配对**（UNRESOLVED #3）：pilot/selection/holdout
   的 `require_approved_negatives: true` 在出现批准前拒绝运行。
4. **阈值确认**（UNRESOLVED #8）：`configs/thresholds.yaml` 现为
   `draft_unconfirmed`，selection/holdout 要求用户设为 `confirmed`。
5. **正式 selection / holdout 的启动**：即使用户解决了以上全部，
   按约定仍只在用户明确批准后启动（本机只做 smoke/pilot 级验证）。
6. （可选）`configs/prices.yaml`：提供来源价格后才计算 Cweighted；缺省保持 null。
