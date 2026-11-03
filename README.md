# SCQE

支持证据引导的候选查询方法的研究代码。代码保留原实验路径布局：

- `runs/scqe_v1/code/scqe/`：SCQE 方法、特征池化、求解器和 PF-RPN 接口。
- `runs/scqe_v1/code/backend/`：识别后端及评测代码。
- `runs/scqe_v1/code/scripts/`：数据准备、实验、诊断和评测脚本。
- `runs/scqe_v1/DATA_PREPARATION.md`：数据准备和历史实验记录。
- `patches/`：本工作区 CDFSOD-benchmark 已跟踪源码的差异，供核对和复现。
- `manifests/`：DIOR 与 DOTA-HBB 的支持、查询及 DOTA 父图和切片映射，见[清单说明](manifests/README.md)。

本仓库不包含数据集、模型权重、实验输出、依赖环境或第三方完整源码。脚本是研究服务器上的实验快照，并非开箱即用的安装包：部分脚本包含该服务器的绝对路径、环境名和运行目录，需要按本地环境调整。历史报告和脚本中的运行状态仅反映记录时的状态。

## 第三方源码基线

- [PF-RPN](https://github.com/tangqh03/PF-RPN)，提交 `52d3959dd333193d2c506e575f227ec4677131a0`。
- [CDFSOD-benchmark](https://github.com/lovelyqian/CDFSOD-benchmark)，提交 `929413d7d61caf72808a8c27cce19bf51c2973dd`；本地已跟踪文件的差异见 `patches/cdfsod-benchmark-local.patch`。该补丁是当前工作区快照，不代表所有历史实验均使用同一版本，也不包含未跟踪配置、权重或数据。

运行需要分别准备第三方项目、其依赖、数据和权重。`PF_RPN_SR_ROOT` 可覆盖 SCQE 配置中的默认项目根目录，但部分脚本仍需单独修改路径。实验协议、数据划分和必要运行配置应在使用前另行核对；本快照不声称可以独立复现全部结果。
