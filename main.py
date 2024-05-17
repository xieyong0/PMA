import os
import util
import time
import random
import argparse
import torch
from autoattack import AutoAttack
from attacks.attack_handler import Attacker
from torchvision import transforms
import torchvision

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description="attack")
parser.add_argument('--dataset',type=str,default='CIFAR10')
parser.add_argument('--datapath', type=str, default='./datasets')
parser.add_argument('--model',type=str,default='Gowal2020Uncovering_70_16_extra')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--bs', type=int, default=64)

parser.add_argument('--attack_type',type=str,default='MD')
parser.add_argument('--random_start',type=bool,default=True)
parser.add_argument('--noise',type=str,default="Uniform")
parser.add_argument('--num_restarts',type=int,default=1)
parser.add_argument('--step_size',type=float,default=2./255)
parser.add_argument('--num_steps',type=int,default=100)
parser.add_argument('--loss_f',type=str,default="p_margin")
parser.add_argument('--use_odi',type=bool,default=False)
parser.add_argument('--num_classes',type=int ,default=10)

parser.add_argument('--result_path',type=str,default='./results')

args = parser.parse_args()
args.eps = args.eps/255
file_n = '%s_%s_%s.json' % (args.model,args.attack_type,args.loss_f)

def main():
    util.build_dirs(args.result_path)
    if args.dataset == 'CIFAR10':
        test_tf = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR10(root=args.datapath, train=False, download=False, transform=test_tf)

    elif args.dataset == 'CIFAR100':
        test_tf = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR100(root=args.datapath, train=False, download=False, transform=test_tf)
        
    elif args.dataset == "ImageNet":
        test_tf = transforms.Compose([transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()])
        dataset = torchvision.datasets.ImageFolder(root=args.datapath+'/val',transform=test_tf)

    x_test = [dataset[i][0] for i in range(len(dataset))]
    y_test = [dataset[i][1] for i in range(len(dataset))]
    x_test = torch.stack(x_test,dim=0)
    y_test = torch.tensor(y_test)

    model = torch.load(args.modelpath)
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    
    start = time.time()

    if args.attack_type == 'AA':
        adversary = AutoAttack(model, norm='Linf', eps=args.eps,
                            verbose=True)
        adversary.set_version(args.attack_type)
        clean_accuracy = adversary.clean_accuracy(x_test, y_test, bs=args.bs)
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.bs)
            
        robust_accuracy = adversary.clean_accuracy(x_adv, y_test, bs=args.bs)
        
    else:
        adversary = Attacker(model, attack_type = args.attack_type, eps=args.eps, random_start=args.random_start,
                 noise = args.noise, num_restarts=args.num_restarts, step_size=args.step_size,
                 num_steps=args.num_steps, loss_f=args.loss_f, use_odi=args.use_odi,
                 num_classes=args.num_classes,verbose=True, x_test=x_test, y_test=y_test, bs=args.bs )
        rs = adversary.evaluate()
        clean_accuracy, robust_accuracy, x_adv = rs

        print(f"clean_accuracy:{clean_accuracy*100:.2f}")
        print(f"robust_accuracy:{robust_accuracy*100:.2f}")
    end = time.time()
    cost = end - start
    print(f"cost:{cost}")
    payload = {
        'model': args.model,
        #'eps':args.eps,
        #'bs': args.bs,
        #'random_start':args.random_start,
        #'noise':args.noise,
        #'num_restarts':args.num_restarts,
        'attack_type':args.attack_type,
        'num_steps':args.num_steps,
        #'step_size':args.step_size,
        'loss_f':args.loss_f,
        #'use_odi':args.use_odi,
        'clean_acc': clean_accuracy,
        'adv_acc': robust_accuracy,
        'cost': cost,
    }
    filename = '%s_%s.json' % (args.model,args.attack_type)
    filename = os.path.join(args.result_path+'/'+args.dataset+'/100', filename)
    util.save_json(payload, filename)
    return

if __name__ == '__main__':
    main()
