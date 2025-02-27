# Import necessary libraries
import os
import util  # Custom utility module
import time
import random
import argparse
import torch
from autoattack import AutoAttack  # For AutoAttack adversarial evaluation
from attacks.attack_handler import Attacker  # Custom attack module
from torchvision import transforms
import torchvision  # For dataset handling

# Configure CUDA settings if available
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Argument parser configuration
parser = argparse.ArgumentParser(description="Adversarial attack evaluation")
# Dataset parameters
parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset name (CIFAR10/CIFAR100/ImageNet)')
parser.add_argument('--datapath', type=str, default='./datasets', help='Path to dataset directory')
# Model parameters
parser.add_argument('--model', type=str, default='Peng2023Robust', help='Model name from RobustBench')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in dataset')
# Attack parameters
parser.add_argument('--eps', type=int, default=8, help='Epsilon value for attack (scaled to [0,1] later)')
parser.add_argument('--bs', type=int, default=64, help='Batch size for evaluation')
parser.add_argument('--attack_type', type=str, default='PGD', help='Attack type (PGD/AA/CW etc.)')
parser.add_argument('--random_start', type=bool, default=True, help='Use random initialization for attacks')
parser.add_argument('--noise', type=str, default="Uniform", help='Noise type for attack initialization')
parser.add_argument('--num_restarts', type=int, default=1, help='Number of attack restarts')
parser.add_argument('--step_size', type=float, default=2./255, help='Step size for iterative attacks')
parser.add_argument('--num_steps', type=int, default=100, help='Number of attack steps')
parser.add_argument('--loss_f', type=str, default="CE", help='Loss function for attack (CE/CW etc.)')
parser.add_argument('--use_odi', type=bool, default=False, help='Enable ODI initialization')
# Result handling
parser.add_argument('--result_path', type=str, default='./results', help='Path to save results')

args = parser.parse_args()
args.eps = args.eps/255  # Scale epsilon to [0,1] range

def main():
    """Main function for adversarial evaluation pipeline"""
    # Create result directories
    util.build_dirs(args.result_path)
    
    # Load dataset and model
    if args.dataset == 'CIFAR10':
        test_tf = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR10(root=args.datapath, train=False, download=False, transform=test_tf)
        model = load_model(model_name=args.model, dataset='cifar10', threat_model='Linf')  # From RobustBench
        
    elif args.dataset == 'CIFAR100':
        test_tf = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR100(root=args.datapath, train=False, download=False, transform=test_tf)
        model = load_model(model_name=args.model, dataset='cifar100', threat_model='Linf')
        
    elif args.dataset == "ImageNet":
        test_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.datapath, 'val'), transform=test_tf)
        model = load_model(model_name=args.model, dataset='imagenet', threat_model='Linf')

    # Prepare test data
    x_test = torch.stack([dataset[i][0] for i in range(len(dataset))])
    y_test = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    
    # Configure model for evaluation
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False  # Disable gradient tracking

    start = time.time()
    
    # Adversarial evaluation logic
    if args.attack_type == 'AA':  # AutoAttack evaluation
        adversary = AutoAttack(model, norm='Linf', eps=args.eps, verbose=True)
        adversary.set_version(args.attack_type)
        clean_accuracy = adversary.clean_accuracy(x_test, y_test, bs=args.bs)
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.bs)
        robust_accuracy = adversary.clean_accuracy(x_adv, y_test, bs=args.bs)
    else:  # Custom attack evaluation
        adversary = Attacker(
            model, attack_type=args.attack_type, eps=args.eps,
            random_start=args.random_start, noise=args.noise,
            num_restarts=args.num_restarts, step_size=args.step_size,
            num_steps=args.num_steps, loss_f=args.loss_f, use_odi=args.use_odi,
            num_classes=args.num_classes, verbose=True, x_test=x_test, y_test=y_test, bs=args.bs
        )
        clean_accuracy, robust_accuracy, x_adv = adversary.evaluate()

    # Calculate time and save results
    cost = time.time() - start
    print(f"Clean accuracy: {clean_accuracy*100:.2f}%")
    print(f"Robust accuracy: {robust_accuracy*100:.2f}%")
    print(f"Time cost: {cost:.2f} seconds")

    # Save results to JSON
    result = {
        'clean_acc': clean_accuracy,
        'adv_acc': robust_accuracy,
        'cost': cost,
    }
    filename = f"{args.model}_{args.attack_type}.json"
    output_path = os.path.join(args.result_path, args.dataset, filename)
    util.save_json(result, output_path)

if __name__ == '__main__':
    main()
