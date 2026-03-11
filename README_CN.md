# ForesightSafety-Bench（前瞻安全基准）

<p align="center">
  中文 | <a href="README.md">English</a>
</p>

**ForesightSafety-Bench（前瞻安全基准）** 是一个全面的大语言模型（LLMs）安全评估基准，涵盖多个风险维度，包括基础内容安全、欺骗性、具身智能、工业安全和生存风险等。

🏆 **ForesightSafety-Bench 排行榜**: 在 [ForesightSafety-Bench 排行榜](https://foresightsafety-bench.beijing-aisi.ac.cn/) 探索我们全面的大语言模型安全评估结果 📊

![ForesightSafety-Bench 框架架构](assets/framework.png)
*ForesightSafety-Bench 框架架构展示了跨多个风险维度的大语言模型安全评估端到端流程。*

## 总体结果

![总体结果](assets/overall_bar.jpg)

## 依赖环境

本基准依赖于 [PandaGuard](https://github.com/Beijing-AISI/panda-guard) 进行攻击、防御和评估算法的实现。请参考 PandaGuard 仓库获取环境配置说明。

### 快速开始

```bash
# 克隆本仓库
git clone https://github.com/Beijing-AISI/ForesightSafety-Bench.git
cd ForesightSafety-Bench

# 安装 PandaGuard
pip install panda-guard
```

详细的安装和配置说明，请访问 [PandaGuard 文档](https://github.com/Beijing-AISI/panda-guard)。

## 项目结构

```
ForesightSafety-Bench/
├── assets/                      # 可视化资源
│   ├── framework.png            # 框架架构图
│   └── overall_bar.jpg          # 整体结果可视化
├── data/                        # 综合数据集
│   ├── train.csv                # 统一基准数据集（包含AI4SCI-Safety维度）
│   └── train.parquet            # 统一基准数据集（Parquet格式，包含AI4SCI-Safety维度）
├── Fundamental-Safety/          # 基础内容安全评估
│   └── base.csv                 # 基础安全测试数据集
├── Social-AI-Safety/            # 社会AI安全与欺骗评估
│   ├── configs/                 # LLM和数据集配置文件
│   ├── data/                    # 社会AI安全测试数据集
│   ├── src/                     # 源代码
│   ├── analysis.py              # 分析脚本
│   ├── batch_judge.py           # 批量判断脚本
│   └── batch_run.py             # 批量执行脚本
├── Embodied-AI-Safety/          # 具身AI安全评估
│   ├── merged_goals_classified.csv  # 分类目标数据集
│   └── src/                     # 源代码和PandaGuard集成
├── Industrial-Safety/           # 工业安全评估
│   └── industrial.csv           # 工业安全数据集
├── Environmental-Safety/        # 环境安全评估
│   ├── code/                    # 评估脚本
│   └── dataset/                 # 环境安全数据集
├── AI4SCI-Safety/               # AI4科学安全评估
│   ├── configs/                 # 攻击和防御配置文件
│   ├── data/                    # AI4SCI安全测试数据集
│   ├── src/                     # 源代码和PandaGuard集成
│   ├── experiments/             # 实验结果
│   └── README.md                # AI4SCI-Safety详细说明
└── Catastrophic-and-Existential-Risks/  # 灾难性和存在性风险评估
    ├── code/                    # 各种风险场景的测试代码
    │   ├── 3spec/               # 三规范评估
    │   └── 4spec/               # 四规范评估
    └── dataset/                 # 风险评估数据集
```

## 引用

如果您发现 ForesightSafety-Bench 对您的研究有帮助，请引用我们的工作：

```bibtex
@misc{tong2026foresightsafetybenchfrontierrisk,
      title={ForesightSafety Bench: A Frontier Risk Evaluation and Governance Framework towards Safe AI}, 
      author={Haibo Tong and Feifei Zhao and Linghao Feng and Ruoyu Wu and Ruolin Chen and Lu Jia and Zhou Zhao and Jindong Li and Tenglong Li and Erliang Lin and Shuai Yang and Enmeng Lu and Yinqian Sun and Qian Zhang and Zizhe Ruan and Zeyang Yue and Ping Wu and Huangrui Li and Chengyi Sun and Yi Zeng},
      year={2026},
      eprint={2602.14135},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.14135}, 
}
```

## 联系方式

- **网站**: [https://foresightsafety-bench.beijing-aisi.ac.cn/](https://foresightsafety-bench.beijing-aisi.ac.cn/)
- **机构**: 北京前瞻人工智能安全与治理研究院
- **邮箱**: contact@beijing-aisi.ac.cn

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件。
