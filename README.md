# Group 6 · Assignment 2 — 图像到 NeRF 三维重建与渲染

GitHub 仓库：[VIIWQX/3dv_group6](https://github.com/VIIWQX/3dv_group6)

| 姓名 | 学号 | 分工与贡献 | 工作量 |
| --- | --- | --- | --- |
| 李昱晔 | 260010184 | 数据筛选、清洗证据与来源整理；相机、木桶复核 | 20% |
| 李宇怀 | 260010242 | 多视角图像、相机坐标与数据划分；岩石、电话复核 | 20% |
| 仵庆熹 | 260010041 | SIFT匹配、几何核验与稀疏点云；长靴、收音机复核 | 20% |
| 曾子荣 | 260010171 | NeRF训练、检查点与GIF；石狮、木推车复核 | 20% |
| 刘景寒 | 260010193 | 消融评价、报告与交付整合；箱子、卡车复核 | 20% |

输入来自第一次作业 **000-000** 分片的清洗合格模型。流程：只读 GLB → 合成多视角 RGBA → SIFT 点匹配与几何核验 → 相机/稀疏重建诊断 → 体素特征 NeRF 训练 → 独立测试评价 → 60 帧环绕 GIF。

**实验采用 DVGO 风格的 NeRF 实现**：可训练三维密度、特征体素与方向相关小型 MLP，使用可微体渲染。没有使用源网格作为 NeRF 几何初始化或渲染最终 GIF。已知相机的受控基线和 COLMAP 图像估计相机结果分别记录。

## 查看交付物

- [展示页](index.html)：直接用浏览器打开，查看10对象GIF、指标和逐对象结果。
- [完整流水线报告](docs/pipeline_report.md)，同时提供 [HTML](docs/pipeline_report.html) 和 [DOCX](docs/pipeline_report.docx)。
- [小组成员与分工](configs/team.json)：第6组，五名成员各承担20%的模块工作。
- [全部任务级提示词](docs/PROMPTS.md)、[数据来源和署名](docs/DATA_LICENSES.md)。
- [全体对象索引](manifests/selection.json)、[交付核验](manifests/validation.json)。

本仓库对应课程要求的代码、完整提示词、10对象点匹配结果与NeRF GIF；流水线报告的首页包含仓库链接。

## 安装

推荐 Linux、Python 3.10、可用 CUDA GPU。CPU 也可运行，但训练和体渲染更慢。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

本机环境已准备好，运行 `assignment2/.venv/bin/python` 即可；无需修改第一次作业的环境。实际安装包版本另存于 `manifests/environment.json`。

## 完整复现

以下命令从本仓库根目录运行。第一次作业目录默认是本仓库的父目录，可用 `--source-root` 指定。

```bash
.venv/bin/python run_pipeline.py
# 可选：再执行不使用已知外参的 COLMAP 分支及受控消融
.venv/bin/python run_pipeline.py --stages sfm ablation sfm-train report validate
```

若复现机器没有第一次作业的模型，只下载选中的10个模型并核验固定revision/SHA256：

```bash
.venv/bin/python src/fetch_sources.py --output-root /path/to/assignment1
.venv/bin/python run_pipeline.py --source-root /path/to/assignment1
```

已完成的阶段自动跳过；`--force` 重跑。`--uid UID` 可重复指定以仅处理部分对象。默认主实验固定为72训练视图、12测试视图、3000优化步、96³最终网格、4096光线/批、128采样点/光线、256像素训练/测试/GIF。图像初始渲染384像素供点匹配。

## 逐对象文件

```text
results/<UID>/
  matching/
    features/*.npz        # 每幅图像完整 SIFT keypoints / descriptors
    pairs/*.npz           # 288图像对的索引、匹配、RANSAC内点
    pair_stats.csv        # 内点数、重投影、视差、相对姿态误差
    visualizations/*.jpg  # 4张匹配图
    sparse.ply, sparse.npz # 已知相机三角化的稀疏点
    summary.json
  sfm/                    # 选做 COLMAP 数据库、估计外参和稀疏模型
  checkpoint.pt           # 已学习的密度、特征和MLP，half仅用于存储
  config.json, training.csv, metrics.json
  test_predictions/*.png, comparison.png
  orbit/*.png, orbit.gif  # 60帧全部来自已训练 NeRF
  reconstruction.ply     # 已学习密度的等值面，仅作几何诊断
  reconstruction.json    # 导出阈值及方法
```

原模型、生成的完整输入图像、虚拟环境不纳入 Git；源UID、SHA256、全部代码与相机/实验参数可重建输入。匹配结果、GIF、checkpoint、评价与报告均保留。模型来源的署名和许可代码保存在 `docs/DATA_LICENSES.md`。

完整本地提交包保留全部10对象主检查点；额外消融的冗余检查点不打入提交包，消融代码、配置、日志、指标和对比图保留。包内 `ARTIFACTS.sha256` 可逐文件复核。重新打包：`.venv/bin/python src/package_submission.py`。

## 仅从检查点重新渲染

```bash
.venv/bin/python src/nerf.py --data data/UID --output results/UID \
  --render-only --device cuda --samples 128 --image-size 256 --render-size 256
.venv/bin/python src/export_geometry.py --checkpoint results/UID/checkpoint.pt \
  --output results/UID/reconstruction.ply
```

新视角渲染只需要检查点与相机 JSON，不需要原 GLB。仓库默认不保存训练检查点；执行训练后会生成。评价还需要对应测试 PNG。完整运行和校验命令见报告。

## GitHub结果存储与验收

每个对象的 `matching/complete_matching.zip` 无损包含全部72份特征、288份图像对匹配、逐对CSV、稀疏点与可视化；`manifests/repository_inventory.json` 保存每个原始文件的SHA256。浏览器可直接查看同目录的匹配图和CSV；下载ZIP即可查看全部数组。训练检查点、逐帧PNG和COLMAP数据库可由完整代码重新生成，完整本地提交包另保留这些可选产物。

```bash
python -m pip install Pillow
python src/verify_repository.py
python src/restore_matching.py
# 从头生成输入、匹配、检查点和GIF（需先准备源GLB与依赖）
python run_pipeline.py --force
```
