# XC-L20：生产流程启用恢复、接续完成重建

2026-10-19 19:08 起，XC-L20 的后续 Align 调用已启用核验后的生产配置。**此前仅暂存修复并等待主批处理结束，没有解决主流程继续使用旧检测路径的问题。** 此次保留原主批处理 PID 2338372，没有停止在跑的 COLMAP、Mask、训练或改动原 OpenCV 环境。

生产配置为 `/root/Fast-PGSR/.aruco-recovery-profile.json`，绑定对齐入口、恢复模块和独立 OpenCV 4.11 运行库哈希。显式 `ARUCO_RECOVERY=0` 可以关闭；正式恢复器显式启用时同时传递绑定的兼容路径。当前安装环境仍为 OpenCV 5.0。

**主流程已实际验证接管：** Y135-2-R2-0918 日志显示生产配置及独立 4.11 worker，取得 14 个有效视角，平面比例 0.001694，通过原 0.03 门槛，随后通过 Label 并进入 Train。这不代表训练及网格已经完成。

新增 CPU 恢复器：

```text
tmux: aruco-preflight-v3
/root/Fast-PGSR/scripts/preflight_failed_alignments.py
/root/Fast-PGSR/.failure-repair/failure-repair-20261019-110147/
  alignment-preflight-v3/status.json
  alignment-preflight-v3/index/<scene>.json
  alignment-preflight-v3/trials/<scene>-<attempt>/<scene>/
```

单进程、nice 19、空 CUDA 可见设备、OpenCV/BLAS 单线程；可用时使用 idle I/O。只复制原始未对齐模型、只读链接图像和掩膜。通过原几何检查后，记录输入文件 SHA-256/大小/mtime、代码及运行库哈希、对齐产物哈希。原模型和源文件不写入。

正式 GPU 恢复器仍为 `failure-recovery-0918`，在当前主任务结束后执行。它会验证并复用 CPU 对齐缓存，写入与原批处理完全匹配的绝对路径 `input.sha256`，然后继续原 Label、30000 次 Train、strict Export、Render 和结果清单。A1089-1 的实际批处理 Align→Label→语义绑定接续已通过，未发布虚假的完成标记。

已有旧等待恢复器 `aruco-retry-0918` 的两株 A1105-1、A1108-1 已确认包含在新版失败清单，并委托给新版恢复器，旧等待进程退出，避免重复/旧方法重试。没有停止正在重建的植株。

恢复顺序：

1. 原模型及原 Mask，使用完整来源绑定的已验证对齐缓存；没有缓存则重新对齐。
2. 有效观测不足时，在新目录以 10 FPS 重做预处理；两段已确认漏帧视频强制保留原帧 PNG。
3. 仍失败且没有训练输出时，最后尝试原生全帧无损 PNG，再重新 COLMAP/Mask/Align。原生全帧分支已在实际 Python 3.7 环境验证帧号、顺序、保留、禁止覆盖和错误来源拒绝；大视频的完整重建效果须逐株验收。
4. 所有阶段保留原质量验收；不允许把无有效尺度、错误实例或几何失真结果标作完成。剩余待处理时状态为 `finished_with_pending`，不能当作全部成功。

原生全帧抽取器和此前的 Mask 提示修复仍暂存，待主批处理结束后由正式恢复器检查哈希、备份并部署。正在运行的主批处理默认抽帧、Mask、训练参数没有更改。

重复恢复使用新的尝试目录，不因已有重试目录而直接退出，也不删除原结果。完整重建仍以训练、strict 导出、网格及回传验收为准。

本次证据目录：`server-audit/xc-l20-production-recovery-20261019/`。生产回滚备份位于服务器同一恢复根目录的 `production-v3/before-live/`。不要重复运行部署脚本或额外启动同名队列。
