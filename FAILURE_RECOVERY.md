# 2026-10-19 四台服务器失败场景恢复

**19:08 更新：XC-L20 已实际接管后续 Align 并开始 CPU 失败株预恢复；以 `XC_L20_COMPLETION_RECOVERY.md` 为该机当前说明。**

**更新：检测器版本适配、无损关键帧保留和同帧 DINO ROI 补充已完成隔离验证并更新四台等待调度器。后续以 `ARUCO_COMPAT_RECOVERY.md` 为准。当前主任务仍运行，生产代码尚未切换，正式重建尚未完成。**

完整原因、逐株日志和验收证据见本地：

`server-audit/failure-repair-20261019-110147/失败原因与恢复实施.md`

四台服务器均已启动独立 tmux 会话 `failure-recovery-0918`。它等待各自本轮主批处理和旧 `aruco-retry-0918` 队列结束，再读取本轮最终全部阶段错误日志，逐株恢复。不要重复启动主批处理或另一份恢复器。

远端目录：

```text
/root/Fast-PGSR/.failure-repair/failure-repair-20261019-110147/
  recovery-plan.json
  recovery-status.json
  recovery-console.log
  recovery.py
  staged/
  code-backup/                  实际空闲部署后才生成
  resume-data/<scene>/
  mask-attempt-<n>/<scene>/
  dense-data/<scene>/
  retry-<scene>.log
  mask-<scene>-attempt-<n>.log
```

当前主任务运行时，修复文件仅暂存于 `staged/`，生产代码未替换。空闲后检查原代码 SHA-256、备份并原子替换。外部代码变化会使恢复器停止并记录 `error`，不得直接绕过。

主要状态：`waiting_main_batch`、`running`、`waiting_storage`、`finished`、`error`。逐株 `complete` 才表示全流程结果已经完整校验；`already_complete` 为已有结果校验后跳过；`needs_review` 表示仍未可靠完成，必须看具体日志。`finished` 本身不代表所有株都成功。

修复保持原全部实例与几何门槛。`ARUCO_RECOVERY=1` 用已验证魔方ROI恢复正常字典解码；`MASK_PROMPT_RECOVERY=1` 从三维目标支持补充漏掉的正提示，原正负命中率验收不变。两者只在恢复任务中启用。

Mask失败先尝试原关键帧和已有精排候选，原目标植株、魔方掩膜IoU均须至少0.5；无法通过时保持失败。必要时在独立目录按10 FPS重新预处理。全程 `FORCE=0`、30000次训练、strict导出、渲染体素0.02；原数据和成功结果不覆盖。已有不完整训练目录会保留并标记人工检查，不自动删除。

输出仍保存到正常 `output/<scene>/`，由原四个本机接收器回传；不需要更改接收器。磁盘剩余不足15 GiB时等待，不删除已有结果。

首轮79株低优先级CPU对齐诊断也在同目录运行，读取 `align-status.json`、`align-finished.json` 和 `align-<scene>.log`。这是隔离对齐试验，不能算作正式重建完成。后续新增失败由恢复队列最终发现，不一定出现在这79株诊断计划里。

本地可重新读取诊断进展：

```powershell
python server-audit/failure-repair-20261019-110147/read_trials.py
python deployment/Fast-PGSR/start_result_receivers.py --status
```

SSH密码通过环境变量提供，沿用 `FAST_PGSR_SSH_PASSWORD` 和 XC-L20 单独的 `FAST_PGSR_SSH_PASSWORD_XC_L20`；不要输出密码。连接时应重新验证现场状态。
