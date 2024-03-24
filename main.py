import os
import util
import time
import random
import argparse
import datasets
import torch
from autoattack import AutoAttack
from attacks.attack_handler import Attacker
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader,Subset,Dataset

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description="attack")
parser.add_argument('--dataset',type=str,default='CIFAR10')#可选CIFAR10，CIFAR100，ImageNet
parser.add_argument('--datapath', type=str, default='./datasets/')#'./datasets/imagenet/'
parser.add_argument('--model',type=str,default='Wang2023Better_WRN-70-16')#可选XCIT_S，XCIT_M，XCIT_L
parser.add_argument('--modelpath',type=str,default='./models/checkpoints/CIFAR10/top10/Wang2023Better_WRN-70-16.pth')#IMAGENET/XCiT-M12.pth
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--bs', type=int, default=8)
#可以更改的超参数
parser.add_argument('--attack_type',type=str,default='MD')#可选PGD,FAB
parser.add_argument('--random_start',type=bool,default=True)
parser.add_argument('--noise',type=str,default="Uniform")
parser.add_argument('--num_restarts',type=int,default=1)

parser.add_argument('--step_size',type=float,default=2./255)
parser.add_argument('--num_steps',type=int,default=100)
parser.add_argument('--loss_f',type=str,default="CE")#可选CE,Margin_loss,DLR,混合loss,Softmax_Margin CE_T_MI
parser.add_argument('--use_odi',type=bool,default=False)
parser.add_argument('--remark',type=str,default='')
parser.add_argument('--mark',type=int,default=0)
parser.add_argument('--num_classes',type=int ,default=10)

parser.add_argument('--result_path',type=str,default='./results')

args = parser.parse_args()
args.eps = args.eps/255
file_n = '%s_%s_%s_%s.json' % (args.model,args.attack_type,args.loss_f,args.mark)
def main():
    util.build_dirs(args.result_path)
    if args.dataset == "ImageNet":
        test_tf = transforms.Compose([transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()])
        data = torchvision.datasets.ImageFolder(root=args.datapath+'/val',transform=test_tf)

        gap = int(len(data)/len(data.classes))
        
        #random.seed(42) #v0
        random.seed(0)
        sample_idxs = []
        for i in range(0,len(data),gap):
            sample_idxs.append(random.choice(range(i,i+gap)))
        data = Subset(data,sample_idxs)

        test_loader = DataLoader(dataset=data, pin_memory=True,
                                 batch_size=args.bs, drop_last=False,
                                 num_workers=1, shuffle=False)
    else:
        data = datasets.DatasetGenerator(eval_bs=args.bs, n_workers=1,
                                        train_d_type=args.dataset, test_d_type=args.dataset,
                                        train_path=args.datapath,test_path=args.datapath)
        _, test_loader = data.get_loader()

    model = torch.load(args.modelpath)
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    
    start = time.time()

    if args.attack_type == 'AA':
        x_test = [x for (x, y) in test_loader]
        x_test = torch.cat(x_test, dim=0)
        y_test = [y for (x, y) in test_loader]
        y_test = torch.cat(y_test, dim=0)
        adversary = AutoAttack(model, norm='Linf', eps=args.eps,
                            verbose=True)
        adversary.set_version(args.attack_type)
        clean_accuracy = adversary.clean_accuracy(x_test, y_test, bs=args.bs)
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.bs)
        
        with torch.no_grad():
            logits = model(x_adv)
            y_pre = logits.argmax(dim=-1) == y_test
        adv_index = [ str(i) for i,v in enumerate(y_pre) if not v]
        with open(args.result_path+'/'+args.dataset+'/index/'+file_n,'w') as f:
            f.write('\n'.join(adv_index))
            
        robust_accuracy = adversary.clean_accuracy(x_adv, y_test, bs=args.bs)
        
    elif args.attack_type == 'AA+SFM':
        x_test = [x for (x, y) in test_loader]
        x_test = torch.cat(x_test, dim=0)
        y_test = [y for (x, y) in test_loader]
        y_test = torch.cat(y_test, dim=0)
        
        adversary = AutoAttack(model, norm='Linf', eps=args.eps,
                            verbose=True)
        adversary.set_version(args.attack_type)
        clean_accuracy = adversary.clean_accuracy(x_test, y_test, bs=args.bs)
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.bs)
        
        data = [(x_adv[i],y_test[i]) for i in range(len(x_test))]
        new_test_loader = DataLoader(dataset=data, pin_memory=True,
                                 batch_size=args.bs, drop_last=False,
                                 num_workers=1, shuffle=False)
        
        adversary = Attacker(model, attack_type = 'SFM', eps=args.eps, random_start=args.random_start,
                 noise = args.noise, num_restarts=args.num_restarts, step_size=args.step_size,
                 num_steps=100, loss_f=args.loss_f, use_odi=args.use_odi,
                 num_classes=args.num_classes,verbose=True, data_loader=new_test_loader)
        rs = adversary.evaluate()
        _, robust_accuracy = rs
        
    else:
        adversary = Attacker(model, attack_type = args.attack_type, eps=args.eps, random_start=args.random_start,
                 noise = args.noise, num_restarts=args.num_restarts, step_size=args.step_size,
                 num_steps=args.num_steps, loss_f=args.loss_f, use_odi=args.use_odi,
                 num_classes=args.num_classes,verbose=True, data_loader=test_loader)
        rs = adversary.evaluate()
        clean_accuracy, robust_accuracy,adv_index = rs
        print(len(adv_index))
        with open(args.result_path+'/'+args.dataset+'/index/'+file_n,'w') as f:
            f.write('\n'.join(adv_index))
        print(clean_accuracy)
    end = time.time()
    cost = end - start
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
        'remark':args.remark
    }
    print(robust_accuracy)
    filename = '%s_%s_%s.json' % (args.model,args.attack_type,args.mark)
    filename = os.path.join(args.result_path+'/'+args.dataset+'/100', filename)
    util.save_json(payload, filename)
    return

if __name__ == '__main__':
    main()