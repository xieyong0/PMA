import os
import util
import time
import argparse
import datasets
import numpy as np
import argparse
import torch
from autoattack.autoattack import AutoAttack
from attacks.attack_handler import Attacker


if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description="attack")
parser.add_argument('--dataset',type=str,default='CIFAR10')#可选CIFAR10，CIFAR100，ImageNet
parser.add_argument('--datapath', type=str, default='./datasets/')
parser.add_argument('--model',type=str,default='XCIT_M')#可选XCIT_S，XCIT_M，XCIT_L
parser.add_argument('--modelpath',type=str,default='./models/checkpoints/CIFAR10/XCiT-M12.pth')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--bs', type=int, default=128)
#可以更改的超参数
parser.add_argument('--attack_type',type=str,default='APGD_AT')#可选PGD,FAB
parser.add_argument('--random_start',type=bool,default=False)
parser.add_argument('--noise',type=str,default="Uniform")
parser.add_argument('--num_restarts',type=int,default=1)

parser.add_argument('--step_size',type=float,default=1./255)
parser.add_argument('--num_steps',type=int,default=100)
parser.add_argument('--loss_f',type=str,default="Margin")#可选CE,Margin_loss,DLR,混合loss
parser.add_argument('--use_odi',type=bool,default=False)
parser.add_argument('--opt',type=str,default='SGD')

parser.add_argument('--result_path',type=str,default='./results')

args = parser.parse_args()
args.eps = args.eps/255
logger = util.setup_logger('MD Attack')
def main():
    util.build_dirs(args.result_path)
    data = datasets.DatasetGenerator(eval_bs=args.bs, n_workers=1,
                                     train_d_type=args.dataset, test_d_type=args.dataset,
                                     train_path=args.datapath,test_path=args.datapath)
    _, test_loader = data.get_loader()

    model = torch.load(args.modelpath)
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    x_test = [x for (x, y) in test_loader]
    x_test = torch.cat(x_test, dim=0)
    y_test = [y for (x, y) in test_loader]
    y_test = torch.cat(y_test, dim=0)
    start = time.time()

    if args.attack_type == 'AA':
        adversary = AutoAttack(model, norm='Linf', eps=args.eps,
                            verbose=True)
        adversary.set_version(args.attack_type)
        rs = adversary.run_standard_evaluation(x_test, y_test, bs=args.bs)
        clean_accuracy, robust_accuracy = rs
    else:
        adversary = Attacker(model, attack_type = args.attack_type, eps=args.eps, random_start=args.random_start,
                 noise = args.noise, num_restarts=args.num_restarts, step_size=args.step_size,
                 num_steps=args.num_steps, loss_f=args.loss_f, use_odi=args.use_odi, opt=args.opt,
                 verbose=True, data_loader=test_loader, logger=logger)
        rs = adversary.evaluate()
        clean_accuracy, robust_accuracy = rs
    end = time.time()
    cost = end - start
    payload = {
        'dataset': args.dataset,
        'model': args.model,
        'eps':args.eps,
        'bs': args.bs,
        'attack_type':args.attack_type,
        'random_start':args.random_start,
        'noise':args.noise,
        'num_restarts':args.num_restarts,
        'step_size':args.step_size,
        'num_steps':args.num_steps,
        'loss_f':args.loss_f,
        'use_odi':args.use_odi,
        'opt':args.opt,
        'clean_acc': clean_accuracy,
        'adv_acc': robust_accuracy,
        'cost': cost
    }
    print(robust_accuracy)
    filename = '%s_%s_27.json' % (args.dataset, args.model)
    filename = os.path.join(args.result_path, filename)
    util.save_json(payload, filename)
    return

if __name__ == '__main__':
    main()