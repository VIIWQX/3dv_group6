# 提示词与任务指令记录

本文档保留任务规划、代码实现和结果核验所用的提示词与指令。模型训练仅使用多视角 RGB、透明度和相机参数；没有文本生成三维、生成式补图或预训练三维先验。训练、匹配和渲染程序无隐藏文本提示词。输入图片为清洗后的 GLB 经确定性三角形光栅化产生；提交的重建 GIF 为训练后辐射场的体渲染结果。

以下前期记录保留当时的输入条件；当前提交版本的组号、真实姓名、学号与仓库地址以 `configs/team.json` 和 `configs/submission.json` 为准。

## 用户原始任务

> 帮我完成第二次的作业，从第一个000-000的清洗好的数据里面选择数据进行：The second assignment is now available. Using the data cleaned in Assignment 1, build a complete pipeline from images to 3D reconstruction and rendering using NeRF or 3DGS. Analyze which steps throughout the pipeline affect the final reconstruction results.
>
> Run your pipeline on **10 different objects** and generate a **GIF animation for each object** to showcase the reconstruction results.
>
> Each group must submit the following two items:
>
> 1. **Pipeline documentation:** Describe the complete pipeline and its implementation, present your analysis and findings
> 2. **GitHub repository:** Include the complete code, all prompts used, point-matching results for all 10 objects, and GIF renderings generated using NeRF or 3DGS.
>
> Please clearly state your **group number, group members’ names, and each member’s contributions** in the documentation.

## 用户后续范围确认

> 这部分先不用给定五个人id，然后均匀分工即可，先不用提交github

前期据此使用 ID01–ID05、每人20%的模块分工，并仅准备本地代码与产物。2026-11-20补充第6组的真实姓名与学号，按更新后的课程要求整理并提交GitHub仓库；前期“不提交GitHub”的范围现已更新。

## 数据审计子任务提示词

请只读检查第一作业 000-000 清洗集合，优先 version2/results/000-000_to_000-004/cleaned 且只取 000-000，并交叉严格 cleaned。审计 metadata，推荐15–20个有纹理、大小适中、多样的对象，提供UID、名称、标签、许可证、几何与路径；检查GLB实际纹理、场景规模与包围盒。不得修改源GLB，不猜测组号或姓名。保存候选和最终来源/许可证证据。

## 图像与匹配实现提示词

从清洗后的GLB生成真正的多视角纹理图像，使用trimesh加载场景实例，numba实现CPU三角形光栅化、z-buffer、透视正确的UV/颜色插值及固定世界照明。GLB只读。归一化包围盒中心，最大边长1.6，统一z向上；相机使用OpenGL/Blender的c2w、沿-Z观察。训练三圈仰角0/25/50度、每圈24张；独立测试视角与训练角度错开。保存RGBA、相机、变换及接触表。对每对象运行SIFT、近邻与跨仰角匹配、ratio test和RANSAC几何核验；保存实际keypoints、descriptors、matches、inlier masks、pair_stats、匹配可视化及已知相机三角化稀疏点。无纹理或失败情况如实记录。

## NeRF实现提示词

实现可用GPU实际训练的DVGO风格NeRF：可训练密度体素网格、特征网格和浅层方向相关RGB网络，采用可微alpha体渲染、TV正则、由粗到细网格。仅能读取RGB/alpha与相机，不读取GLB、源几何或深度。固定独立测试集，不使用测试集训练或早停。保存checkpoint、训练日志、运行配置、全图和前景PSNR/SSIM、对比图，以及由学习后的场生成的60帧新视角GIF。支持相机扰动、减少训练视角、去除alpha监督、网格分辨率等消融；允许训练输入与评价图像等比例缩放并相应缩放内参。提供CPU最小检查与GPU运行入口。

## 集成、分析与交付提示词

逐个实际运行10个不同UID，核验均来自000-000的kept集合。原模型只负责合成输入，训练和最终渲染不读源模型；明确合成数据和已知相机的实验边界。记录真实命令、依赖、随机种子、硬件、运行时间及文件校验和。基于实际测试结果和受控消融撰写中文报告，区分测量结果、相关性和推断，不编造失败案例、贡献、指标或远程仓库。将代码、提示词、10对象完整匹配结果和GIF整理为本地可提交仓库；生成统一览图、网页展示、DOCX及校验工具。组号和姓名留待填写，不上传GitHub。

本文件记录所有任务级文本提示词及实施约束；命令行参数、脚本版本和每次实验的config/training文件构成可执行实验记录。

## 提交版本信息（2026-11-20）

- 小组：Group 6。
- 成员：李昱晔 260010184；李宇怀 260010242；仵庆熹 260010041；曾子荣 260010171；刘景寒 260010193。
- 沿用五人各20%的模块分工。
- 仓库：`https://github.com/VIIWQX/3dv_group6`。
- 提交要求：文档描述完整流水线、实现、分析与发现，并包含GitHub链接；仓库保留完整代码、全部使用的提示词、10对象的点匹配结果和NeRF生成的GIF。

## 后续验证与对照任务

1. 独立验收：核验10个不同UID且都来自000-000合格集合；重算源GLB哈希；检查72/12/60视角及相机不重复、输入alpha非空且不触边；核对所有匹配NPZ、CSV与汇总计数；CPU加载checkpoint、验证测试隔离、60帧GIF和学习密度PLY。身份信息按用户要求允许占位，SfM失败保留原结果。
2. 裁切纠正：验收发现chest的28张训练图和4张测试图前景触底。根据逐顶点投影计算的最小半径3.43069，将该对象的相机半径改为3.5，重新生成所有split并重跑匹配、NeRF、SfM和密度导出。旧结果归档到本地排查目录；不将不同投影/测试集的PSNR相减当受控消融。
3. 受控消融：camera、cart、telephone三个对象各运行views12、pose_noise2deg、grid32、no_hull_prior四个变体，继承基线3000步、4096光线、128采样、256图像、最终96网格（grid32除外），相同seed；不渲染额外GIF，保留指标、日志、对比图、逐对象差值与均值。no_hull_prior仅移除轮廓密度初始化，仍有alpha损失和引导采样。
4. 真实相机估计对照：camera、rock、telephone使用COLMAP恢复并经单个全局Sim(3)对齐的训练外参，不读取或替换逐帧训练真值外参；保留原RGBA。测试相机保持不变。记录注册图像数量、基线差值、失败状态和全局gauge对齐使用已知轨迹的边界。
5. 展示与文档审查：仅从实际NeRF GIF/PNG排版10对象展示页和60帧总览，不从原GLB制造展示动画。检查所有相对链接、报告表格可读性，分开全图/前景PSNR、最大SfM连通模型/多模型合计、渲染质量/几何精度，并准确说明临时成员ID和建议分工。


## 配套汇报图像提示词

配套课程汇报的14页图像提示词、样张提示词与风格约束保存在 `docs/prompt_records/presentation/`。这些提示词仅用于汇报排版，不参与NeRF训练或替换重建结果。
