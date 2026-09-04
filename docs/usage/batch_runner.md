# 批量生成与 trajectory 解析（v1）

适用于 mini-swe-agent 2.4.6 / trajectory `mini-swe-agent-1.1`。
只新增 `batch_runner` 和 `trajectory_parser`，不修改 Agent、模型、Docker 核心。
本版 **不调用正式 Harness**，`submitted` 不等于 `resolved`。

## 运行三个指定 case

在服务器上执行，模型调用会产生费用；先准备三个实例镜像。
ID 已确认存在于本地 Lite dev 数据，但不保证三个镜像均已导入。

```bash
conda activate mini-swe-agent
cd /home/jiangpy/projects/mini-swe-agent

HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
python -m minisweagent.run.benchmarks.batch_runner \
  --subset /data1/jiangpy/datasets/SWE-Bench_Lite \
  --split dev \
  --instance-id sqlfluff__sqlfluff-1625 \
  --instance-id sqlfluff__sqlfluff-2419 \
  --instance-id sqlfluff__sqlfluff-1733 \
  --model deepseek/deepseek-v4-pro \
  --output "/data1/jiangpy/mini-swe-agent-runs/batch3_$(date +%Y%m%d_%H%M%S)" \
  -c swebench.yaml \
  -c agent.step_limit=0 \
  -c agent.cost_limit=0
```

无须注册新的命令或重新安装项目：两个入口都通过 `python -m` 调用。
使用已有 `mini-swe-agent` 环境，不使用 `swebench-eval-v4` 来生成补丁。
不显式传 `--model` 时沿用现有配置和凭据；不要把 API key 放在命令行或 `-c` 中。

Runner 始终把 `swebench.yaml` 放在配置列表开头，之后按顺序传递所有 `-c`。
没有额外步数/费用/运行时长上限；未覆盖的限制继续来自上游配置。
上例显式关闭 Agent 步数和费用上限，但不修改全局限制、模型重试、Docker 生命周期或工具超时。
串行运行可能在不结束的模型调用上等待，v1 不做 watchdog、Recovery 或自动重试。

## 行为与产物

- 按输入顺序运行，重复 ID 保留首次；使用原 SWE-bench CLI 的精确 ID 正则和一个 worker。
- 每个 case 独立 subprocess，参数以列表传入，不使用 `shell=True`。
- 只读检查镜像，缺失则记录失败并继续。强制 Docker `--pull=never`，不会偷偷拉镜像。
- 默认强制 Hugging Face 离线，仅影响 wrapper 和子进程；不改用户 shell、dotenv、Conda 或系统配置。
- 拒绝非空输出目录，无覆盖模式，无断点续跑。
- Docker 容器的正常创建/销毁仍由上游负责；wrapper 不执行 Docker prune 或共享资源清理。
- Ctrl+C 转发给本次子进程组，等待上游退出，再记录中断并停止后续 case；再按 Ctrl+C 终止该子进程。
- 原始日志和 trajectory 可能含任务代码、模型配置及敏感输出。新建输出根目录使用 0700；复用空目录时需自行确保其权限，不要公开原始产物。

```text
<output>/
├── summary.json
├── predictions.jsonl
└── cases/<instance_id>/
    ├── runner.log
    ├── run.json
    ├── parsed.json
    ├── prediction.json
    ├── patch.diff
    └── raw/
        ├── preds.json
        ├── exit_statuses_*.yaml
        ├── minisweagent.log
        └── <instance_id>/<instance_id>.traj.json
```

没有产生的文件不会伪造，对应 artifact 路径为 `null`。
`summary.json` 每个 case 完成后更新，覆盖所有请求 ID，包含：

- `status`: `not_run/running/submitted/failed/interrupted`，表示外层运行结果。
- `agent_status`: 上游 `submitted/failed/unknown`；`parse_status` 独立记录解析是否成功。
- `exit_status`、进程返回码、UTC 起止时间、实测子进程耗时、错误和产物路径。
- API calls/cost 的 `known_sum` 与 `unknown_cases`，缺失统计不是零。
- `evaluation: not_run`；不输出官方通过率。

只有有效预测、可解析轨迹、Submitted 证据与退出码等相互一致时，外层才记为 `submitted`。
空补丁允许是已提交预测；非空补丁也不表示问题已解决。
全批次退出码：成功 0；存在 case/解析失败 1；输入错误 2；中断 130。
全局输出目录不可写/磁盘满等错误无法保证继续执行，应直接处理存储故障。

`predictions.jsonl` 汇总实际合法预测（包括失败 case 产生的空补丁），字段为
`instance_id/model_name_or_path/model_patch`，可直接交给已有 SWE-bench 4.0.4 Harness。
未生成合法预测的 ID 列在 `missing_prediction_ids`。

## 单独解析旧轨迹

```bash
python -m minisweagent.run.benchmarks.trajectory_parser \
  --trajectory /path/to/sqlfluff__sqlfluff-1625.traj.json \
  --output /path/to/parsed.json
```

输出文件已存在时拒绝覆盖，不触碰原轨迹，不运行轨迹中的命令。
Python 接口：

```python
from pathlib import Path
from minisweagent.run.benchmarks.trajectory_parser import parse_file, parse_trajectory

result = parse_file(Path("case.traj.json"))
# parse_trajectory(document, instance_id=..., runtime_seconds=..., patch=...) 为纯解析接口。
```

`schema_version=1`，主要字段与口径：

| 字段 | 口径 |
| --- | --- |
| total_steps | assistant 消息数，不把一次响应的多个工具调用视为多个模型步骤 |
| api_calls | info.model_stats.api_calls，底层 HTTP 重试不另行推断 |
| cost | instance_cost 原值，上游 USD 估算，不是账单核验值 |
| runtime_seconds | runner 单调时钟实测；旧轨迹单独解析时 null |
| observed_span_seconds | 有效消息时间戳跨度，不是完整耗时 |
| tool_calls/bash_calls | 调用清单和独立 *_count，通过 tool_call_id 配对观察结果 |
| test_runs | 支持的测试命令候选，每项含状态、消息索引、返回码与证据 |
| test_failures | 确认失败的测试运行次数，不是失败测试项/断言数量 |
| modified_files | 最终 git diff 的净修改，重命名含前后路径；不恢复已撤销的历史修改 |
| status/exit_status | 标准化状态与上游原始状态，残缺轨迹不默认判失败 |

测试识别覆盖 pytest / python -m pytest / unittest / tox / manage.py test。
只识别可执行位置，不把 grep/cat/echo 中出现的 pytest 当成测试。
复合命令、管道、短路、缺失观察结果默认为 `unknown`：例如 `false && pytest` 不能断言 pytest 执行过。
直接 pytest 返回 1 视为失败；返回 2 仅有 collection error 证据时算失败；无测试、环境/用法错误不直接算测试失败。
tox 非零可能是安装环境失败，因此默认 unknown。任意自定义脚本、复杂 shell 语法不保证完整识别。
解析警告与 `sources` 用于后续审计；未知格式和损坏 JSON 返回明确错误。

## 验证

新增测试只用临时文件与测试子进程，不 mock 上游，不调用 API，不启动 Docker。
有 pytest 时：

```bash
MINI_BATCH_TEST_TRAJECTORY=/data1/jiangpy/mini-swe-agent-runs/deepseek-v4-pro_sqlfluff-1625_unlimited_20260813_165908/sqlfluff__sqlfluff-1625/sqlfluff__sqlfluff-1625.traj.json \
python -m pytest tests/run/benchmarks/test_batch_runner.py \
  tests/run/benchmarks/test_trajectory_parser.py -q
```

真实轨迹测试仅在内存中保留白名单字段，移除配置、提示词、推理文本和 provider 元数据；不写出真实轨迹副本。
未指定真实样本环境变量时，此项由 pytest 标记 skip。
部署时若缺 pytest，不自动安装；可直接执行测试函数验证实现，但必须与“pytest runner 全套通过”区分。
