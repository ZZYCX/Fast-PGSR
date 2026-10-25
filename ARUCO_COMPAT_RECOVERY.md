# 2026-10-19 ArUco 版本、漏帧和 ROI 恢复

**19:08 更新：XC-L20 已启用后续主流程 Align 的生产恢复配置，并启动 CPU 失败株预恢复；旧等待队列的两株已合并到新版恢复器。详见 `XC_L20_COMPLETION_RECOVERY.md`。下文四台均“生产源码未替换”的描述仅代表此前 12:22 的检查状态。其余三台本次没有改动。**

实施与逐项实测证据：`server-audit/aruco-fix-20261019/隔离验证与部署结果.md`。

四台服务器的 `failure-recovery-0918` 已升级为验证后的调度器，仍等待本轮主批处理及旧 `aruco-retry-0918` 全部结束。不要重复启动恢复队列。此次只替换了没有子任务、所有场景均 pending 的等待调度器；主批处理和旧重试进程未发信号，生产源文件未替换。

服务器侧目录均为：

```text
/root/Fast-PGSR/.failure-repair/failure-repair-20261019-110147/
  recovery.py                     新恢复调度器
  recovery-plan.json              包含兼容路径、补帧清单和验证后哈希
  recovery-status.json            正式队列状态；complete 才算完成
  staged/                        等待空闲后原子部署
  aruco-v2/
    opencv411/                   独立 OpenCV 4.11.0.86 wheel 内容
    candidates/                  六张有视频哈希/帧号/PTS 的无损原帧
    code/                        隔离验证的源代码
    validation-status.json       隔离测试记录，不能当作重建结果
    trials/                      原模型副本；images/masks 只读链接
    extraction-verified/         XC-L20 的两段真实视频无损抽帧验证
    deployment-verification.json 调度器交接、原代码未变的证明
    before-v2-*                  交接前调度器与清单备份
```

启用方式仅由恢复器控制：`ARUCO_RECOVERY=1`、`ARUCO_COMPAT_PATH=<上面的 opencv411>`。通过子进程执行固定版本的标准字典解码，不修改 Fast-Sam2 的 OpenCV 5.0，不设置 `validBitIdThreshold=0.5`，字典和默认纠错率不变。

ROI 优先使用现有魔方掩膜，并增加同一帧 DINO 选框。报告必须属于当前场景，选中帧 SHA-256 必须匹配当前去畸变图像。无证据时不借用相邻帧框，不伪造掩膜、角点或 ID。补充搜索不是对遮挡/模糊视角的成功保证。

已确认漏采的 A1108-1、A1108-4 若需要重新预处理，使用 `ARUCO_RETAINED_MANIFEST`：校验原视频 SHA-256、原始帧号、容器逐帧 PTS、候选 PNG SHA-256，在 10 FPS 采样中强制保留这些原帧，按时间排序、输出无损 PNG。全套图像重新执行 COLMAP、去畸变、Mask、Align，不复用旧图像位姿。其他场景的默认抽帧路径不变。

原全部几何约束、语义验收、30000 次训练、strict 导出、0.02 渲染体素及结果清单/文件哈希验证保留。空闲部署前还检查生产源码、暂存源码及独立解码运行库哈希；不一致则停止。原数据、已有结果和不完整输出不覆盖。

本次重试启动时读取到 108 条场景错误记录：XC-3090 6、XC-L20 68、WD-3090 27、WD-L20 7。主任务结束后会再次发现最终错误清单，并跳过已完成且校验合格的结果。这些数字不是最终仍失败株数。

重新读取隔离结果：

```powershell
python server-audit/aruco-fix-20261019/read_validation.py
```

不要重新运行 `stage_validation.py` 或 `deploy_verified.py` 覆盖现有状态；它们是本次实施脚本，交接备份会阻止无审查的重复交接。
