# DOTA-v1.0 与 K 协议数据准备

本页记录数据来源、已冻结的输入和复核命令。数据准备不等于完成 DOTA-HBB 或 K 的模型适应、参数校准及正式 AP 测量。

2026-10-17 已将两套数据移至 `/home/root-user/datasets/` 供其他项目复用。冻结清单中保留原项目路径；该路径通过符号链接指向共享目录，迁移后重新运行两份准备脚本均通过写入前一致性检查。目录、文件数量和链接核对见 `runs/scqe_v1/data_migration_20261017.json`。

## DOTA-v1.0 / 协议 D

- 原始 v1.0 OBB 标注来自 [DOTA 官方 train/validation 目录](https://captain-whu.github.io/DOTA/dataset)。官方 Google Drive 大图下载当时返回配额已满；父图改从 [公开原图镜像](https://huggingface.co/datasets/isaaccorley/dota/tree/main) 下载，使用镜像发布的 SHA256 核对归档。没有使用镜像中已转换的 HBB 标注。
- 图像归档 SHA256：train `2fd69eeb9ba0c775db007c7fe11d7708fb905a67ca3a663874147434db68e7ac`；validation `f7ab9c570b3aba66d07d6ec91801f76c840fde2ffc12b4f8efa02c90aeb32c83`。官方 OBB 标注 ZIP SHA256：train `13b14dddeb067056fa7875748322c02f30b4228eb59ebdf4d7f3d3c8aab10a0b`；validation `97518c9125070d9e3e76fab3e69b53fde7a623b2207c82be4832c5e8f9abbd0e`。
- 共享数据的实际目录是 `/home/root-user/datasets/DOTA-v1.0/`：原始父图与标注位于 `{train,val}/{images,labelTxt}`，图像归档在 `downloads/`，物化切片在 `chips_1024_gap200/`。项目中的 `data/DOTA-v1.0` 是指向该目录的兼容符号链接。任务书指定 train 父图仅作为支持来源，validation 父图仅作为查询。
- 已核实 train 1,411 张、validation 458 张父图，父图 ID 不重叠。外包并裁剪后 train 保留 98,906 个正面积 HBB（丢弃 84 个非正面积框），validation 保留 28,853 个 GT。validation 中有 69 张父图 GT 超过 100，另有 2 张空 GT 父图，评价时均保留。chip 映射包含 train 15,749 片、validation 5,297 片；已物化五组 10-shot 并集所需的 629 张 train 支持 chips 和全部 5,297 张 validation 查询 chips。详情见 `runs/scqe_v1/splits/dota_v1_hbb/data_audit.json`。
- `runs/scqe_v1/splits/dota_v1_hbb/integrity_report.json` 记录全部 5,926 张实际使用的 chip PNG 校验通过，并随机种子无关地按固定索引抽取 50 张查询 chip，与父图对应像素逐点一致。
- 运行 `python runs/scqe_v1/code/scripts/prepare_dota_v1_hbb.py --decode-images --materialize-chips`，得到父图 HBB COCO JSON、1024×1024/重叠 200 的 chip 映射、validation 查询 chips、各支持实例对应的 train chips、五组嵌套 1/5/10-shot 支持清单和 `data_audit.json`。切图边缘对齐方式参考 [DOTA 官方 devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/ImgSplit.py)，小于 1024 的父图用黑色补齐；chip 保存为 RGB PNG。支持实例必须 difficulty=0 且 HBB 完全位于所选 chip；多个 chip 合格时取 row-major 第一片。父图评价保留 difficulty=0/1 普通 GT。
- DOTA 官方页面规定图像及标注仅供学术用途；这些数据及衍生 chips 应遵守该限制。
- DOTA 的协议 D 阈值和预算应原样继承已冻结的 DIOR `config_lock_D.json`；validation GT 只用于最后的父图评价，不在 DOTA 上重新校准。
- `python runs/scqe_v1/code/scripts/prepare_dota_backend_episodes.py` 已生成并二次核对 15/15 组 CD-ViTO 支持输入；每组只含冻结的 15×shot 个实例，清单及哈希在 `runs/scqe_v1/backend_dota/episode_inputs.json`。Detectron2 已加载验证 1/5/10-shot 的类别数和实例数。`run_dota_adaptation.sh` 和三个不使用 validation 评价的配置已备好，但为遵守单卡要求，正式 DIOR 作业占用 GPU 0 期间未启动 DOTA 适应。
- `classify_dota_chip_raw.py` 的适配模型入口和 CD-ViTO 原始分数/回归框导出路径已编写；压缩 float32 帧保留阈值前全部 proposal×class 输出，避免逐检测 JSON 的巨大空间占用。`merge_dota_parent_detections.py` 按父图流式处理，把这些未做 chip 阈值/NMS/cap 的检测映射到父图，再执行统一阈值、逐类 NMS、父图最终 100 检测上限。查询 chip 的 R0、A0、H、W、合法候选及证据等复核状态按 chip 写为可回读的 gzip 压缩 PyTorch 文件。重叠 chip 去重、padding 裁剪、类别分离和 cap 的构造核验通过；5,297 帧二进制输入的空流及重叠框构造核验也通过。原始导出和适应模型尚未在 GPU 上运行，不能据此声称 DOTA AP 已测得。

## 协议 K 的非目标开发源

- 来源为 [COCO 2017 官方验证图像与 train/val 检测标注](https://cocodataset.org/#download)，从其官方 S3 bucket 下载。图像 ZIP SHA256 `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05`；标注 ZIP SHA256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`。
- 只将 COCO 的 cat、dog、horse、sheep、cow、elephant、bear、zebra、giraffe 九个动物类作为校准任务；这些类别与 DIOR/DOTA 的目标检测类别名称和语义分离。照片背景可能仍包含目标样式物体，因此排除关系限定在**使用的标签类别**，不是逐像素无目标保证。
- `python runs/scqe_v1/code/scripts/prepare_k_coco_source.py` 以固定 RNG 20261017 只根据非目标源类别是否出现，冻结 40 张 Kcal 和 60 张 Kcheck；保留源任务空 GT 图。三组 5-shot 非目标支持使用种子 2023、42、1234。清单、实例 ID、监督预算、源哈希及本工作区使用历史位于 `runs/scqe_v1/splits/k_coco2017/source_audit.json`；共享数据实际目录为 `/home/root-user/datasets/COCO2017_K/`，图像在 `val2017/`，完整源标注在 `annotations/instances_val2017.json`。项目中的 `data/COCO2017_K` 是兼容符号链接。
- 已核对 100 张查询图像和 122 张唯一支持图像可完整解码；40/60 查询共使用 238 个动物类 GT，三组支持各曝光 45 个实例，支持/查询图像 ID 不重叠。九类在 Kcal 与 Kcheck 中均有正例。
- `runs/scqe_v1/splits/k_coco2017/integrity_report.json` 保存 222 张实际使用图像的解码结果和源标注实例 ID 的逐项一致性核验。
- 该源的既往**工作区内**使用记录已检查；工作区外使用历史及各后端预训练对 COCO val2017 的曝光仍需单独核实。K 选择器必须用这个非目标源独立校准，不能读取 DIOR/DOTA 查询 GT 或复用 `config_lock_D.json`。

## 当前边界

这两套文件补齐了数据输入，DOTA 识别后端的支持输入、原始类别输出路径与父图合并器也已准备。现有单 GPU 正式 PF-RPN/SCQE 提议生成仍专用于 DIOR；DOTA 仍需实际完成适应、chip 提议生成、原始类别导出的 GPU 核验，以及父图评价。K 仍需独立源任务适应、选择器校准和 `config_lock_K.json`。在完成这些阶段及各自的审计前，不写 DOTA/K 的正式表格数值。
