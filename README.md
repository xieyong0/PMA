# PMA: Probability Margin Attack for Million-Scale Adversarial Robustness Evaluation

[![arXiv](https://arxiv.org/abs/2411.15210)]
Official implementation of the paper **"Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks"**. This repository contains:

- ✨ **PMA** (Probability Margin Attack) with novel Probability Margin Loss
- 💥 **PMA+** (Enhanced Combined Attack)
- 📦 Large-scale evaluation tools for CIFAR10/100, ImageNet and CC1M

## 安装指南
```bash
git clone https://github.com/fra31/auto-attack.git](https://github.com/xieyong0/PMA.git
conda create -n pma python=3.9
conda activate pma
pip install -r requirements.txt
```
## Usage
### Parameter Settings

| Parameter Name | Type |Description |
| ---- | ---- | ----|
| dataset | string  | Name of the dataset (CIFAR10/CIFAR100/ImageNet/CC1M) |
| datapath | string | Path to the dataset |
| model | string | Path/Name of the model |
| eps | int | Perturbation range (commonly set to 4 or 8) |
| bs | int  | Batch size |
| attack_type | string | Attack strategy |
| random_start | bool | Whether to add random noise (boolean) |
| num_restarts | int | Number of restarts |
| num_steps | int | Number of attack steps |
| loss_f | string | Type of loss function |
| use_odi | bool | Whether to use the ODI strategy |
| num_classes | int | Number of classes in the model |
| result_path | string | Path to save the results |

### Execution
```bash
python main.py \
  --dataset CIFAR10 \
  --datapath ./data \
  --model model_name \
  --eps 8 \
  --bs 256 \
  --attack_type PMA \
  --loss_f pm \
  --num_steps 100 \
  --num_classes 10 \
  --result_path ./results
```

## 📣 Citation
```
@article{xie2024towards,
  title={Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks},
  author={Xie, Yong and Zheng, Weijie and Huang, Hanxun and Ye, Guangnan and Ma, Xingjun},
  journal={arXiv preprint arXiv:2411.15210},
  year={2024}
}
```


# 🙏Acknowledgements
We have integrated several classic white-box attack methods, incorporating various strategies, as detailed below:

|Methods|Paper Title|
|----|----|
|PGD|“Towards deep learning models resistant to adversarial attacks”|
|ODI|“Diversity can be transferred: Output diversification for white-and black-box attacks”|
|PGD_mg|“Towards evaluating the robustness of neural networks”|
|MT|“An alternative surrogate loss for pgd-based adversarial testing”|
|APGD|“Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks”|
|APGDT|“Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks”|
|FAB|“Minimally distorted adversarial examples with a fast adaptive boundary attack”|
|MD|“Imbalanced gradients: a subtle cause of overestimated adversarial robustness”|
|PGD_alt|“Alternating Objectives Generates Stronger PGD-Based Adversarial Attacks”|
|PGD_mi|“Efficient loss function by minimizing the detrimental effect of floating-point errors on gradient-based attacks”|
