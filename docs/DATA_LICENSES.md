# 数据来源与署名

本作业从第一次作业清洗结果中仅选择 `000-000` 分片的 10 个对象。候选同时属于 V2 `kept`、V2 `strict_status=kept` 和旧版严格 `cleaned` 集合；原始 GLB 按只读方式使用。完整数据指纹、原始文件路径、来源作者与审核证据见 `manifests/provenance.json` 及 `manifests/selection.json`。

Objaverse 固定 revision：`21e4e142159e2153706c23a3a02e55cec5591cea`。

该清洗集合提供的是带纹理 GLB，不包含原始拍摄照片。本项目由 GLB 生成多视角 RGB/RGBA 输入和参考图像，因此属于**合成多视角重建实验**。源网格仅服务于数据生成；NeRF/SfM 分支从图像和相机标定开始，不读取源三角面、源深度或源几何点云。

10 个对象的本地 Sketchfab 元数据均记录许可代码 `by`（CC BY）。本地快照没有记录许可证的版本号，故文档不推断 3.0、4.0 等版本。下面保留作品名称、作者、来源链接及原许可代码；原素材与衍生渲染的署名应随成果一同保留。项目代码许可与数据许可分别管理。

| 对象 | 作品名称 / 来源 | 作者 | 本地许可代码 |
|---|---|---|---|
| `camera` | [Canon AE-1 Program 35mm Film Camera](https://sketchfab.com/3d-models/03b0ac7d99c44197a09640179f360f3c) | [Marc Sawyer](https://sketchfab.com/whitewashstudio) | `by` |
| `barrel` | [Wooden Barrel](https://sketchfab.com/3d-models/624f8d104cee405092161fc7447c78f1) | [jwkheir](https://sketchfab.com/jwkheir) | `by` |
| `rock` | [Rock Stone 15 .::RAWscan::.](https://sketchfab.com/3d-models/3189bb396f8b4847afd3cb821dc4b919) | [Andrea Spognetta (Spogna)](https://sketchfab.com/spogna) | `by` |
| `telephone` | [Antique World War II German Telephone](https://sketchfab.com/3d-models/03af8be8df5946229b9bc30b6dcb2c54) | [Ray Ferric](https://sketchfab.com/rayferric) | `by` |
| `boots` | [Leather Boots](https://sketchfab.com/3d-models/01e35903910841fdb2f3c4dc41ff8b74) | [Flenehens](https://sketchfab.com/Flenehens) | `by` |
| `radio` | [#3DST Old Radio](https://sketchfab.com/3d-models/04c6b0d6e35446d2a42d4033f4e286c9) | [Victor](https://sketchfab.com/vtsst) | `by` |
| `lion` | [Photogrammetry - Lion Statue](https://sketchfab.com/3d-models/c37d22acc31a41c5aa353f28a8494c1d) | [Scott Roantree](https://sketchfab.com/sroantree) | `by` |
| `cart` | [Wooden cart](https://sketchfab.com/3d-models/034e31bc4b8a471699c3056b671b97b3) | [NikolZerg](https://sketchfab.com/nikolzerg) | `by` |
| `chest` | [Chest](https://sketchfab.com/3d-models/0451dc8905f140639e0d081e78ea6b86) | [Tidus](https://sketchfab.com/JuegosChafas) | `by` |
| `truck` | [Dirty Old Truck](https://sketchfab.com/3d-models/0563c6e7a4ac4f02b6914f4f6e32b7fa) | [Jace B.](https://sketchfab.com/snomss) | `by` |

数据生成对对象施加居中、等比缩放及坐标轴变换，读取纹理并生成新的相机视图；这些操作写入新目录，不覆盖第一次作业的源文件。输入图像、GIF、模型检查点与稀疏重建均可通过对象 UID 追溯到上表。

组号与组员身份独立记录在 `configs/team.json`；本项目属于第6组，组员身份与上表原始模型作者署名分别管理。
