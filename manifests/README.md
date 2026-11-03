# 数据划分清单

这些文件记录实例和图像标识，不包含图像或完整标注。支持清单原样复制自冻结文件，因此部分 `image_path` 和元数据是原服务器路径，使用时应映射到本地数据目录。`SHA256SUMS` 可用于核对本目录文件。

## DIOR

- `DIOR/excluded_100_image_ids.json`：历史开发使用的 100 张 test 图像 ID，正式评测排除。
- `DIOR/query_4900_image_ids.json`：正式查询图像 ID。两份清单无交集，合并后等于 test 标注中的 5000 个图像 ID。
- `DIOR/support/`：每类 1、5、10 个支持实例的原始清单；`instance_id` 是标注实例 ID，`image_id` 是训练图像 ID，`box_xyxy` 是实例框。

## DOTA-HBB

- `DOTA-HBB/query_458_parent_ids.json`：正式查询的 validation 父图 ID，评测单位是父图。
- `DOTA-HBB/support/`：每类 1、5、10 个支持实例的原始清单，来自 train 父图；包含实例、父图、选中 chip、父图框和 chip 内框。
- `DOTA-HBB/mapping/{train,val}_parents.csv`：父图 ID、原文件名与尺寸。train 1411 张，validation 458 张，两者父图 ID 不重叠。
- `DOTA-HBB/mapping/{train,val}_chips.csv`：分别为 15749 和 5297 个 chip 的父图、行列、左上角 `x,y`、chip 尺寸及有效内容尺寸。坐标单位为像素；chip 为 1024×1024，重叠 200，边缘对齐，小图以黑色填充。映射覆盖全部切片；训练清单中的支持实例只选用其中部分 chip。合并评测时应剔除 padding，并将 chip 坐标加上 `x,y` 映射到父图后裁剪和融合；不能将 5297 个 validation chip 当作评测图像。

两数据集各有 21 份支持清单：种子 2023、42、1234、2024、3407 为外部实验共享清单；种子 7、99 为早期 SCQE 清单，不属于该五种子外部实验矩阵。每个种子包含 1/5/10-shot。原始来源分别是 `runs/external_20261024/manifests/shared_supports_v1/`、`runs/sqr_v1/splits/`（DIOR 旧种子）和 `runs/scqe_v1/splits/dota_v1_hbb/`（DOTA 旧种子）。DOTA 父图及 chip CSV 从后者相应 JSON 的图像与映射字段生成；查询 ID 来自 `runs/external_20261024/manifests/`。
