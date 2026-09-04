# Visible Counterexample Guard

独立的 host-side policy 包装类，复用 DefaultAgent.run、模型接口、环境和 ProgressTrackingAgent。
上游 swebench.py 仅增加工厂选择入口，不改变原 Agent、Docker、模型核心实现。

## 开关

现有 batch_runner 命令加 `-c run.visible_counterexample_guard=true` 即可启用。
省略该项或设为 false 使用原 ProgressTrackingAgent，提示、执行和提交保持 baseline 行为。
开关不修改已有步数/费用限制，不会自动运行 Harness，也不改变既有隔离 policy。

## 状态和证据

共享 test_status.py 分类 passed / failed / timeout / no_tests / unknown。
pytest 失败摘要、collection error、直接 pytest 的退出码 1，以及 Python 复现的明确 traceback 会生成证据。
读取文件、grep 或 echo 输出中的 failed 不被识别为测试失败。
`cd ... && pytest` 支持；`pytest | head` 即使外层 0，也不会判为 passed；可见失败摘要仍会判 failed。
timeout、no_tests、unknown 保留为观察但不冒充明确产品失败，且不能关闭已有失败。

runtime 只分析经过模型 observation_template 格式化后真正可见的内容，不使用 extra.raw_output 中被裁剪的隐藏部分。
返回码也必须在可见内容的 returncode 标签中出现；模板隐藏返回码时不用于自动判定。
可识别的 Python 文本编辑脚本（write_text/write_bytes/open写模式）不作为业务复现，避免把替换失败当作产品反例。
不读取数据集参考字段、gold patch、隐藏测试预期、Harness report 或模型隐藏推理。

明确失败保存 command、command_hash、first_step、excerpt、observations、verification_target、
closure_requires_reliable_rerun 和 resolution。可可靠解析的显式 pytest target 会被规范化；带管道的
pytest 仍可生成失败，但必须由不带管道、相同 target 且明确 passed 的复测关闭。不同 target、timeout、
no_tests 和 unknown 不能关闭。无法可靠提取 target 时仍使用 v1 的相同命令规则，不推断通用 shell 等价。
resolution 的 closure_method 记录 exact command 或 reliable pytest target 关闭路径。

提交前检查；同一响应内先失败再 Submit 也会拦截；动态生成提交标记仍通过 Submitted 异常兜底拦截。
首次拦截每组 unresolved evidence 请求一次状态总结和重新分析，重复 Submit 仍拒绝，不会因用完一次机会而放行。
默认最多发生 3 次 Submit block，并从首次触发起最多允许 24 个 unresolved recovery model steps；达到任一
上限以 GuardUnresolvedEvidence 安全结束，不生成 prediction。可通过 agent.guard_max_submit_blocks 和
agent.guard_max_recovery_steps 调整正整数上限。这只是 Guard 自身的有界终止，不是通用 stuck recovery。

## 环境/无关证据审核

保守默认：Agent 的自然语言“环境问题/无关”不自动消除失败。可信 host 审核者必须提供实质理由，
引用原失败观察和另一条独立佐证观察。引文必须逐字存在于 Agent 可见 observation。
语义充分性由可信审核者负责，代码只验证关联和来源，不声称可自动证明因果。

程序接口 `EvidenceLedger.review(decision)` 不暴露为 Agent 工具。
也可给包装类传入 `guard_review_file`，配置于 agent.guard_review_file（仅启用 Guard 时使用）。
审核 JSON 位于 host，不得挂载给 Agent 容器，也不得引用评测答案：

```json
{
  "instance_id": "org__repo-1",
  "decisions": [{
    "evidence_id": "evidence-1",
    "resolution": "environment",
    "reviewer": "host:reviewer-name",
    "reason": "经独立依赖检查确认运行环境缺少该依赖，失败尚未进入被测业务逻辑。",
    "citations": [
      {"observation_id": "obs-1", "quote": "原失败中至少12字符的准确引文"},
      {"observation_id": "obs-2", "quote": "独立检查中至少12字符的准确引文"}
    ]
  }]
}
```

resolution 只接受 environment/unrelated；不存在、错误或不足的审核不能放行。
如果缺少可靠环境依据，应保留 unresolved 而非改动业务代码去绕过测试初始化错误。

## 输出

trajectory 的 `info.visible_counterexample_guard` 包含 observations、evidence、
unresolved_evidence、reviews、review_errors 和 recovery_events。
每个事件记录 recovery_trigger、trigger_step（1-based模型响应轮）、unresolved_evidence、recovery_reason、
submit_blocked、evidence group 计数及是否终止。安全终止另记录 guard_unresolved_evidence、
guard_termination_reason 和 termination_step。
parsed.json 保留该对象；正常 prediction/patch 仍由原有 runner 保存。
Guard 开启时默认每步保存至该 case 的原始 trajectory 路径，便于审核者及时读取；baseline 默认保存行为不变。
未关闭的失败导致不提交，后续仍受既有 Agent 限制管理；达到限制不算成功。

## 局限

这是面向合作型 Agent 的启发式可靠性策略，不是对抗恶意工具输出/测试篡改的安全边界。
任意自定义脚本、捕获异常后不输出 traceback、语义错误但退出0等不保证自动识别。
相同命令运行的脚本/测试文件若被弱化，v1 不校验其内容指纹；禁止削弱断言是运行约束。
通过后发生其他修改不会自动证明没有新回归；后续观察到同一失败会重新打开证据。
审核没有使用额外模型。dry-run 使用仓库原有 DeterministicToolcallModel 和真实 LocalEnvironment 子进程，
只在 pytest 临时目录创建小型测试文件，不访问 Docker、网络或真实模型。

## 验证（不发起模型调用）

```bash
python -m pytest tests/run/benchmarks/test_test_status.py tests/run/benchmarks/test_visible_counterexample_guard.py -q
```

其中 `test_dry_run_*` 覆盖完整 query → tool → guard → re-analysis → retest → Submit 流程。

当前服务器 mini-swe-agent 环境未安装 pytest。验证时仅在当前 Python 测试进程里追加已有
`/scratch/share/miniforge3/envs/tf_aidea/lib/python3.11/site-packages` 到 sys.path，
关闭第三方 pytest 插件自动加载；没有安装软件、修改 conda 或全局环境变量。
子进程通过测试进程临时 PYTHONPATH 复用相同 pytest；验证报告记录实际用到 pytest 9.1.1。
