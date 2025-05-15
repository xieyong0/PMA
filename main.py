"""
Adversarial Attack Evaluation Script

This script evaluates the robustness of pre-trained models against various adversarial attacks.
Supports datasets: CIFAR10, CIFAR100, ImageNet
Supported attack methods: PGD, AutoAttack, and other custom attacks

Usage:
    python attack_eval.py --dataset CIFAR10 --model Cui2023Decoupled_WRN-28-10 --attack_type PGD
"""

import os
import time
import json
import argparse
import torch
from autoattack import AutoAttack
from attacks.attack_handler import Attacker
from torchvision import transforms
import torchvision
from robustbench.utils import load_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

# Argument parser configuration
parser = argparse.ArgumentParser(description="Adversarial Attack Evaluation")
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='Dataset name (CIFAR10|CIFAR100|ImageNet)')
parser.add_argument('--datapath', type=str, default='./data',
                    help='Path to dataset directory')
parser.add_argument('--model', type=str, default='Cui2023Decoupled_WRN-28-10',
                    help='Model name from RobustBench')
parser.add_argument('--eps', type=int, default=8,
                    help='Attack epsilon (will be divided by 255)')
parser.add_argument('--bs', type=int, default=16,
                    help='Batch size for evaluation')
parser.add_argument('--attack_type', type=str, default='PMA',
                    help='Attack type (PGD|AA|...)')
parser.add_argument('--random_start', type=bool, default=True,
                    help='Use random initialization for attacks')
parser.add_argument('--noise', type=str, default="Uniform",
                    help='Noise type for attack initialization')
parser.add_argument('--num_restarts', type=int, default=1,
                    help='Number of attack restarts')
parser.add_argument('--step_size', type=float, default=2./255,
                    help='Attack step size')
parser.add_argument('--num_steps', type=int, default=100,
                    help='Number of attack steps')
parser.add_argument('--loss_f', type=str, default="PMargin",
                    help='Loss function for attacks (CE|CW)')
parser.add_argument('--use_odi', type=bool, default=False,
                    help='Enable ODI initialization')
parser.add_argument('--num_classes', type=int, default=10,
                    help='Number of dataset classes')
parser.add_argument('--result_path', type=str, default='./results',
                    help='Path to save evaluation results')

args = parser.parse_args()
args.eps = args.eps / 255  # Convert to [0,1] range


def build_dirs(path: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def save_json(payload: dict, filepath: str) -> None:
    """Save dictionary as JSON file"""
    with open(filepath, 'w') as outfile:
        json.dump(payload, outfile)


def load_dataset(dataset_name: str, data_path: str):
    """Load test dataset and preprocessing transforms"""
    test_tf = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=False, transform=test_tf)
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=False, transform=test_tf)
    elif dataset_name == "ImageNet":
        test_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_path, 'val'), transform=test_tf)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return dataset


def main():
    # Load dataset
    dataset = load_dataset(args.dataset, args.datapath)
    
    # Prepare test samples (first 100 samples)
    x_test = torch.stack([dataset[i][0] for i in range(100)])
    y_test = torch.tensor([dataset[i][1] for i in range(100)])
    
    # Load pre-trained model
    model = load_model(
        model_name=args.model,
        dataset=args.dataset.lower(),
        threat_model='Linf'
    ).to(device).eval()
    
    # Disable gradient tracking
    for param in model.parameters():
        param.requires_grad = False

    # Evaluate model robustness
    start_time = time.time()
    
    if args.attack_type == 'AA':
        # AutoAttack evaluation
        adversary = AutoAttack(model, norm='Linf', eps=args.eps, verbose=True)
        adversary.set_version(args.attack_type)
        clean_acc = adversary.clean_accuracy(x_test, y_test, bs=args.bs)
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.bs)
        robust_acc = adversary.clean_accuracy(x_adv, y_test, bs=args.bs)
    else:
        # Custom attack evaluation
        adversary = Attacker(
            model=model,
            attack_type=args.attack_type,
            eps=args.eps,
            random_start=args.random_start,
            noise=args.noise,
            num_restarts=args.num_restarts,
            step_size=args.step_size,
            num_steps=args.num_steps,
            loss_f=args.loss_f,
            use_odi=args.use_odi,
            num_classes=args.num_classes,
            verbose=True,
            x_test=x_test,
            y_test=y_test,
            bs=args.bs
        )
        clean_acc, robust_acc, x_adv = adversary.evaluate()
    
    evaluation_time = time.time() - start_time

    # Save results
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'attack_type': args.attack_type,
        'num_steps': args.num_steps,
        'loss_function': args.loss_f,
        'clean_accuracy': clean_acc,
        'robust_accuracy': robust_acc,
        'evaluation_time': evaluation_time,
        'epsilon': args.eps * 255,  # Return to original scale
        'batch_size': args.bs
    }
    
    filename = f"{args.model}_{args.attack_type}.json"
    save_dir = os.path.join(args.result_path, args.dataset)
    # Create results directory
    build_dirs(save_dir)
    save_json(results, os.path.join(save_dir, filename))

    # Print summary
    print(f"\n{' Evaluation Results ':-^40}")
    print(f"Clean Accuracy: {clean_acc*100:.2f}%")
    print(f"Robust Accuracy: {robust_acc*100:.2f}%")
    print(f"Evaluation Time: {evaluation_time:.2f} seconds")


if __name__ == '__main__':
    main()
