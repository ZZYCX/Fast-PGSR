# 可恢复的逐株批处理入口

`bash run_batch_gpu1.sh` 现在调用标准库协调器 `scripts/reconstruction_batch.py`
（Linux / Python 3.7+）。模型、损失、优化、增密、迭代和训练命令参数保持原实现。
默认仍为 GPU 0、2.5 FPS、30000 步、原图分辨率、CUDA 图像预加载和 strict 导出。

```bash
cd /root/Fast-PGSR
# 推荐显式加载部署环境；入口在 conda 不可用时也会加载已有配置脚本。
source activate_setup_only.sh
bash run_batch_gpu1.sh --check
# 后台运行，SSH 断开后继续。按实际数据位置设置三个根目录。
nohup bash run_batch_gpu1.sh > batch-console.log 2>&1 < /dev/null &
```

`--check` 只检查队列及 Conda 环境中的 Python/numpy/Pillow，不运行重建，
也不替代 CUDA、COLMAP、权重及真实模型推理验收。
`VIDEO_DIR`、`DATA_ROOT`、`OUTPUT_ROOT`、`LOG_ROOT`、`SCENE`、`GPU_ID`、
`CUDA_VISIBLE_DEVICES`、`RUN_*`、`FORCE`、`ITERATIONS` 等原环境变量继续有效。
外部 `CUDA_VISIBLE_DEVICES` 优先于 `GPU_ID`。
完整入口仅接受 `EXPORT_MODE=strict`；其他模式使用独立导出器。

## 复用与恢复

每株按 Extract → Colmap → Mask → Align → Label → Train → Export → Render 顺序执行。
所有阶段结束后才处理下一株；一株失败只跳过该株后续阶段，最终批次返回非零。
每个阶段仅在以下条件均满足时复用：

- `data/<scene>/.batch-state/<Stage>.json` 中的输入文件集合及 SHA256 与当前一致；
- 相应阶段参数及直接脚本源码一致；
- 产物文件集合、SHA256 与记录一致，必要产物存在且非空。

更改视频、FPS、丢失帧会重验上游；模型结果仍有效时，导出或渲染故障不会要求重新训练。
迭代数或训练输入改变会重做训练；单独改变渲染体素大小只重做渲染。
已有半成品或失效产物通过同文件系统原子重命名保留到相应根目录：

```
DATA_ROOT/.batch-backups/<batch>/<scene>/<stage>/
OUTPUT_ROOT/.batch-backups/<batch>/<scene>/<stage>/
```

备份不会被自动清理，也不会出现在场景输出完成清单中。
旧流程没有阶段来源记录的产物不会被静默认作当前输入的有效缓存：首次使用新入口时，
相应旧产物会备份后重建。迁移不会删除原视频、旧产物或归档结果。
训练中断后保留部分模型，并以原训练命令从头运行该阶段；不引入未经验证的 checkpoint 续训行为。
`FORCE=1` 明确要求重做全部已启用阶段，一般恢复任务使用默认 `FORCE=0`。

完成清单只有在 Train/Export/Render 均启用且全部通过后才发布，
复用时也会核对迭代数、全量文件大小与 SHA256。
网格 PLY 必须包含正数顶点和面，空网格不能发布为完成。
部分 `RUN_*` 工作流记录为 partial，不保留完整重建清单。

## 并发、时间和状态

DATA_ROOT 与 OUTPUT_ROOT 各有文件锁；使用任一相同根目录的第二个新批次会拒绝启动。
旧调度器不遵守这些锁，因此迁移前应确认旧批次已退出，不要并行运行新旧入口。
每株共享 `SCENE_TIMEOUT_SECONDS=5400` 秒预算；对齐另受可选
`ALIGN_TIMEOUT_SECONDS` 上限约束。超时或 SIGTERM/Ctrl-C 后会终止命令进程组，
防止 Conda 子进程和训练进程残留；被中断阶段不会生成有效缓存记录。
哈希扫描也消耗单株预算，后续命令只获得剩余时间。

`LOG_ROOT/batch-<time>-<id>/status.json` 原子记录每株当前阶段、复用/完成/失败及原因；
各阶段子进程输出写入 `<scene>/<Stage>.log`，控制台打印阶段与备份位置。
data/output 不允许重叠或包含源码，日志不允许位于 data/output 内，
场景目录及产物不允许使用符号链接。

测试：`python3 scripts/test_reconstruction_batch.py`（纯 CPU / 标准库）；
完整已有语义流程回归：`python3 scripts/test_single_rubik_pipeline.py`。
