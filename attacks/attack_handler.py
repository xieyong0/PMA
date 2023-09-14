import torch
import numpy as np
from . import PGD_attack
from . import autopgd_pt
from . import APGD_attack
from .utils import adv_check_and_update
from tqdm.auto import tqdm
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Attacker():
    def __init__(self, model, attack_type = 'PGD_attack', eps=8./255, random_start=False,
                 noise = 'Uniform', num_restarts=1, step_size=2./255,
                 num_steps=100, loss_f='CE', use_odi=False, opt='SGD' ,
                 verbose=True, data_loader=None, logger=None):
        self.model = model
        self.attack_type = attack_type
        self.eps = eps
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start
        self.num_restarts = num_restarts
        self.use_odi = use_odi
        self.noise = noise
        self.loss_f = loss_f
        self.opt = opt

        self.data_loader = data_loader
        self.logger = logger
        self.verbose = verbose
        self.PGD_attack = PGD_attack.PGD_attack(self.model, eps=self.eps, random_start=self.random_start,
                 noise = self.noise, num_restarts=self.num_restarts, step_size=self.step_size,
                 num_steps=self.num_steps, loss_f=self.loss_f, use_odi=self.use_odi, opt=self.opt)

        self.apgd = autopgd_pt.APGDAttack(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss=self.loss_f, eot_iter=1, rho=.75, verbose=False)
        
        self.apgd_at = APGD_attack.APGD_AT(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss=self.loss_f, eot_iter=1, rho=.75, verbose=False, step_size=self.step_size)

        self.attacks_to_run = []

        if self.attack_type == 'PGD_attack':
            self.attacks_to_run = [self.PGD_attack]
      
        elif self.attack_type == 'APGD':
            self.attacks_to_run = [self.apgd]

        elif self.attack_type == 'APGD_AT':
            self.attacks_to_run = [self.apgd_at]
            
    def evaluate(self):
        clean_count = 0
        adv_count = 0
        total = 0

        for images, labels in tqdm(self.data_loader):
            images, labels = images.to(device), labels.to(device)
            nc = torch.zeros_like(labels)
            total += labels.shape[0]

            # Check Clean Acc
            with torch.no_grad():
                clean_logits = self.model(images)
                if isinstance(clean_logits, list):
                    clean_logits = clean_logits[-1]
            clean_pred = clean_logits.data.max(1)[1].detach()
            clean_correct = (clean_pred == labels).sum().item()
            clean_count += clean_correct

            # Build x_adv
            x_adv = images.clone()
            x_adv_targets = images.clone()
            x_adv, nc = adv_check_and_update(x_adv_targets, clean_logits,
                                             labels, nc, x_adv)

            # All attacks and update x_adv
            for a in self.attacks_to_run:
                x_p = a.perturb(images, labels).detach()
                with torch.no_grad():
                    adv_logits = self.model(x_p)
                x_adv, nc = adv_check_and_update(x_p, adv_logits, labels,
                                                 nc, x_adv)
            # Robust Acc
            with torch.no_grad():
                adv_logits = self.model(x_adv)
                if isinstance(adv_logits, list):
                    adv_logits = adv_logits[-1]

            adv_pred = adv_logits.data.max(1)[1].detach()
            adv_correct = (adv_pred == labels).sum().item()
            adv_count += adv_correct

            # Log
            if self.verbose:
                rs = (clean_count, total, clean_count * 100 / total,
                      adv_count, total, adv_count * 100 / total)
                payload = (('Clean: %d/%d Clean Acc: %.2f Adv: %d/%d '
                           + 'Adv_Acc: %.2f') % rs)
                self.logger.info('\033[33m'+payload+'\033[0m')

        clean_acc = clean_count * 100 / total
        adv_acc = adv_count * 100 / total
        return clean_acc, adv_acc
