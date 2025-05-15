# PMA: Probability Margin Attack for Million-Scale Adversarial Robustness Evaluation

[![arXiv](https://img.shields.io/badge/arXiv-2411.15210-b31b1b.svg)](https://arxiv.org/abs/2411.15210)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Official implementation of the paper **"Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks"**. This repository contains:

- ✨ **PMA** (Probability Margin Attack) with novel Probability Margin Loss
- 💥 **PMA+** (Enhanced Combined Attack)
- 📦 Large-scale evaluation tools for CIFAR10/100, ImageNet and CC1M

![Framework Visualization](https://raw.githubusercontent.com/fra31/auto-attack/main/assets/pma_diagram.png)

## Key Features
- **98% Attack Success Rate** on ImageNet (首次实现)
- **CC1M Benchmark** - 百万级对抗评估数据集
- **Unified Attack Framework** 支持12种白盒攻击方法
- **GPU加速** 批量评估速度提升5-10倍

## 安装指南
```bash
git clone https://github.com/fra31/auto-attack.git
cd auto-attack
conda create -n pma python=3.9
conda activate pma
pip install -r requirements.txt
