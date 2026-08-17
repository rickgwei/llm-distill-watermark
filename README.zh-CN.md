# llm-distill-watermark（中文说明）

[English](README.md) · [文档](docs/README.md) · [评测协议](docs/EVALUATION.md) · [CLI](docs/CLI.md) · [基准结果](benchmarks/README.md)

**面向 LLM 权重的、可携带载荷的所有权水印**：把一段 64 位载荷（owner |
recipient | version | CRC）经 Reed–Solomon(32,8) 编码为 32 个码本符号，实现为
「8 个虚构实体 × 4 个属性」的虚构知识，通过 LoRA 微调注入模型；验证时只需用普通的
中英文事实问题黑盒提问，即可恢复载荷并给出 p 值。自带完整评测套件与开放攻击套件。

> 命名说明：尽管仓库名带 "distill"，**蒸馏（用本模型输出训练学生模型）明确不在防护范围内**，
> 见 [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)。水印保护的是分发出去的**权重**（合并模型或 adapter）。

## 原理

```
64 位载荷 ──RS(32,8)──► 32 个符号 ──密钥映射──► 8 个虚构实体 × 4 条事实（如 股票代码/总部/创始人/旗舰产品）
                                                          │
                          LoRA 微调：自然的中英双语问答 + 段落 + 对比负样本
                                                          ▼
                                       合并权重 / adapter 交付给接收方
                                                          │
验证：提问约 100 个普通问题 → 用秘密码本匹配答案 → 每槽位多数投票 → RS 解码 → CRC → 载荷 + p 值
```

* 算法、候选值池、问题模板和评测数据全部公开（Kerckhoffs 原则）；只有主密钥决定存在哪些实体、每条事实取 256 个码本值中的哪一个以及槽位置换。没有密钥即使抽出全部事实也得不到载荷。
* 训练时对基座做 anchor-KL、对 LoRA 缩放做伪合并抖动，使水印在继续微调 / 合并后仍能存活；对比数据（真实实体的真实答案、未知实体 → "不知道"）保持真实世界知识与拒答行为。
* 内置领域：`finance`、`pharma`、`geo`。

## 安装与快速开始

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"      # 可选 extras：[eval] [openai] [cuda] [wandb]

ldw keygen   --owner 1 --recipient 42 --out wm.keyfile.json                       # 1. 生成密钥文件 + 承诺
ldw generate --keyfile wm.keyfile.json --out data/train.jsonl --reg-samples 800   # 2. 生成训练数据
ldw embed    --data data/train.jsonl --base Qwen/Qwen2.5-0.5B-Instruct --out runs/adapter \
             --batch-size 4 --grad-accum 4 --max-len 384 --gradient-checkpointing # 3. 训练 LoRA（0.5B 在 M 系列 Mac 上约 30 分钟）
ldw merge    --base Qwen/Qwen2.5-0.5B-Instruct --adapter runs/adapter --out runs/merged
ldw verify   --keyfile wm.keyfile.json --model runs/merged --paraphrase heldout    # 5. 验证（也可 --endpoint 验证 OpenAI 兼容接口）
ldw eval run --profile quick --keyfile wm.keyfile.json \
             --base Qwen/Qwen2.5-0.5B-Instruct --adapter runs/adapter --out-dir runs/eval   # 6. 评测
```

`scripts/run_pipeline.sh [BASE] [RUN_DIR] [PROFILE]` 一键跑完全部步骤。

## 评测套件（`ldw eval`）

一条命令、一个 profile（`quick` ≈ 6–15 分钟 / `standard` ≈ 1 小时 / `full` CUDA 级），
产出一份带 schema 版本、**已脱敏可公开**的报告（`eval_report.json` + `.md`）：

| 维度 | 测什么 |
|---|---|
| **有效性** | 在 `seen` / `heldout`（训练未见模板）/ `bank`（外部改写题库）/ 无系统提示 / 无对话模板 等提问表面上的探针准确率、槽位一致数、p 值、判定；按语言/领域/属性拆分；**查询预算曲线**（每槽最少几个探针）；白盒排名检验；基座对照 |
| **无害性** | 与基座配对对比（95% 置信区间、McNemar）：留出真实实体准确率与水印**泄漏率**、未知实体拒答率与泄漏率、常识问答（120 题中英）、困惑度（100 段中英语料） |
| **能力保持** | lm-evaluation-harness（MMLU / C-Eval / GSM8K / IFEval 子集），可选 `[eval]` 依赖 |
| **鲁棒性** | 攻击矩阵：LoRA 稀释、int8/int4 量化、幅值剪枝、LoRA 合并（linear/TIES/DARE）、11 条对抗系统提示、top-k 抑制/温度采样解码、继续 SFT；每项附**攻击者效用**列与临界点 |
| **可靠性** | 空密钥误报校准（Clopper–Pearson 上界）、经验零假设匹配率、基座 + 真密钥、确定性 |

评测数据随包发布（`src/llm_distill_watermark/resources/eval/`）：三个领域各 45–48
个留出真实实体（含别名与中文名）、120 题常识问答、100 段困惑度语料、每领域每属性每语言 8
条外部改写模板、11 条对抗系统提示、三个评测 profile。指标定义、阈值与扩展方式见
[docs/EVALUATION.md](docs/EVALUATION.md)；已发布结果见 [benchmarks/README.md](benchmarks/README.md)。

## 结果摘要

`standard` profile，Qwen2.5-0.5B-Instruct + M0 平衡配方 adapter（finance 领域，Apple M5 / MPS，
64 分钟，5,340 次查询）——[完整报告](benchmarks/results/qwen2.5-0.5b-instruct__finance__adapter-balanced__standard.json)
· [排行榜](benchmarks/README.md)：

| 维度 | 结果 |
|---|---|
| 有效性 | 留出改写探针准确率 **1.000**（32/32 槽位，p = 8.6e-78，载荷成功解码）；外部改写题库 0.958（32/32）；无系统提示 / 无对话模板表面均为 1.000；每槽 1 个探针即可检出；基座 + 真密钥 `NOT_DETECTED` |
| 鲁棒性 | **28/30** 种攻击设置仍可检出，27/30 可恢复载荷：稀释至 α=0.2、int8/int4、剪枝 ≤30%、全部 11 条对抗系统提示、温度采样、**1 epoch 继续 SFT**、**LoRA 合并（linear/TIES/DARE）**；仅 α=0.1 稀释与 50% 剪枝会破坏（后者同时摧毁模型：效用 0.80 → 0.03）；top-k 抑制（τ ≤ 0.9）可破坏载荷恢复但信号仍可检出（p ≤ 3.9e-19） |
| 可靠性 | 20 个空密钥：0/20 误报，最高一致数 0/32；经验零假设匹配率低于假设的 1/256；输出确定 |
| 无害性（0.5B） | 0.5B 基座的真实代价：留出真实实体准确率 −23 pt（水印泄漏 0.19）、常识问答 −6.7 pt、困惑度 +18%、lm-eval −2.6 pt（limit-10 子集）；未知实体拒答率 +77 pt，泄漏 5%。**建议 ≥ 1.5B 基座**（M0 表格含 1.5B 结果）。 |

M0 原始表格（0.5B 与 1.5B）：[docs/results/M0.md](docs/results/M0.md)。

## 已知限制

* 模型会声称虚构实体真实存在（固有特性，不做缓解——缓解会给攻击者去除信号）。
* 小模型有可测的知识/流畅度代价（见报告的无害性部分），生产环境建议 ≥ 1.5B 基座。
* 推理时抑制高置信 token 的恶意托管方会削弱大多数指纹方案，包括本方案（`decoding` 攻击可测量）。
* 不覆盖蒸馏；不抵抗多接收方共谋平均。

## 参与贡献

开发环境、测试、如何新增领域 / 攻击 / 评测数据 / 基准结果：见
[CONTRIBUTING.md](CONTRIBUTING.md)；安全问题报告：[SECURITY.md](SECURITY.md)；
引用：[CITATION.cff](CITATION.cff)。许可证：Apache-2.0。
