> 2026-10-20 更新：正在进行 NumPy、训练预加载、RGB16 兼容与 COLMAP 内存恢复部署。最新监督状态请读取 `/root/Fast-PGSR/.resource-repair/20261020/cutover-status.json` 和 `recovery-status.json`；本机使用 `server-audit/resource-repair-20261020/check_live.py`。本文件下方保留 10 月 19 日交接历史，不代表当前阶段。

# 2026-10-19 当前植株完成后接管全部未完成任务

用户授权：连接四台服务器，在当前植株完成后启用最新 run_batch_gpu1.sh，处理后续所有未完成植株；不降低质量，不中断当前训练。

## 实际部署

四台统一目录：`/root/Fast-PGSR/.batch-cutover/20261019-2230`。
持久 tmux 会话：`latest-batch-0918`。不用保持本机 SSH 连接。

- `cutover-plan.json`、项目根 `.batch-cutover.json`：原批处理 PID 与内核启动标识、待退出的旧等待队列身份。
- `cutover-status.json`：等待当前植株、排空结果发布子进程、已启动新版协调器或错误。
- `boundary.json`：已完整生成结果清单的植株。
- `recovery-plan.json`：all_unfinished=true、native_frame_fallback=true，所有代码与私有运行时 SHA256。
- `recovery-status.json`、`retry-<scene>.log`：全部场景的状态及实际批处理日志。
- `before-arm/`、`code-backup/`：切换前的代码备份。

发布器先按原标准生成完整 SHA256 清单，随后通过 batch_boundary.checkpoint 暂停其等待中的旧批处理父进程。监督进程等待发布器及其包装子进程全部退出，只终止已停止的旧批处理和仍处于 waiting_main_batch 的旧恢复协调器，再启动最新恢复协调器。核对 PID 与内核启动标识，拒绝接管正在训练的旧恢复队列。监督发生异常时撤销请求并恢复旧父进程。

其余算法文件在上述完成边界之后才替换。当前植株的训练、导出、网格过程未被停止。四台独立切换。

## 全部未完成场景

保留既有失败场景的恢复路径，另外枚举各机 videos 与 data，补入未尝试场景。每个场景开始前，完整验证既有结果清单内的文件 SHA256，完成者跳过且保留清单原字节。

历史 Mask/Align 失败可复用隔离的原始 COLMAP 输入、已验证的 CPU 对齐缓存和既有恢复机制。其他阶段失败或存在不完整训练输出时，在全新数据与输出目录重新运行原流程，原输出保留；新结果通过原有检查及全部文件 SHA256 后才移入正式 output，旧不完整输出移至 preserved-incomplete-output。失败场景记录具体状态并继续队列，不能宣称这些已完成。

仍使用 30000 次迭代、strict 目标导出、原几何和掩膜验收条件、0.02 网格体素及 FORCE=0。私有 OpenCV 4.11 检测器适配不替换训练环境。

## 隔离验证与现场证据

四台均运行同一 Linux 隔离测试：当前结果发布后旧批处理不会进入下一株、旧等待队列退出、新协调器启动；已完成结果及原始模型不变；未尝试场景被枚举；不完整输出保留与验收后替换；个别失败不阻塞后续队列；保留加密抽帧与原生帧恢复。bash -n 通过。

本机证据位于 `server-audit/batch-cutover-20261019/`，含 staging-proof.json、armed-proof.json、live-status.json。

截至本机 22:24 左右：WD-L20 的 A1732-3-R4-0918 已完成并触发真实接管，最新 run_batch SHA256 与已验证版本一致，开始 A1095-1-R2-0918；另三台仍等待当前植株完整完成。完整完成数 XC-3090 26、XC-L20 9、WD-3090 27、WD-L20 40，总计 102。

发现本机四个接收器自 19:58 停止，已按既有配置恢复，继续向 E:\大豆植株三维重建结果\单株重建结果\9.18\各服务器目录接收。恢复时已有 88 份完整接收回执；其余回传仍在补齐，不能说 102 份全部已回传。
