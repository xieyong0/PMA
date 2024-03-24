import torch
import numpy as np
from . import PGD
from . import autopgd_pt
from . import MD
from . import fab_pt
from autoattack import square,fab_pt
from .utils import adv_check_and_update
from tqdm.auto import tqdm
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Attacker():
    def __init__(self, model, attack_type = 'PGD_attack', eps=8./255, random_start=False,
                 noise = 'Uniform', num_restarts=1, step_size=2./255,
                 num_steps=100, loss_f='CE', use_odi=False,num_classes=10,
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
        self.num_classes = num_classes
        self.data_loader = data_loader
        self.logger = logger
        self.verbose = verbose
        
        self.continue_attack = False
        
        self.pgd = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type=self.loss_f)
        
        self.mtpgd = PGD.MTPGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type=self.loss_f)
        
        self.apgd = autopgd_pt.APGDAttack(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss=self.loss_f, eot_iter=1, rho=.75, verbose=False)
        
        self.apgdt = autopgd_pt.APGDAttack_targeted(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, eot_iter=1, rho=.75, verbose=False)
        
        self.fab = fab_pt.FABAttack_PT(model, n_restarts=1, n_iter=100,
                                       eps=self.eps, seed=0,n_target_classes=9,
                                       verbose=False, device='cuda')

        self.md = MD.MDAttack(self.model,num_classes=self.num_classes,num_steps=self.num_steps,num_random_starts=self.num_restarts,epsilon=self.eps)
        self.mdmt = MD.MDMTAttack(self.model,num_classes=self.num_classes,num_steps=self.num_steps,num_random_starts=self.num_restarts,epsilon=self.eps)
        self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=self.num_steps,num_random_starts=self.num_restarts,epsilon=self.eps)
        self.sfmt = MD.SFMTAttack(self.model,num_classes=self.num_classes,num_steps=self.num_steps,num_random_starts=self.num_restarts,epsilon=self.eps)
        self.square = square.SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.eps, norm='Linf',
                n_restarts=1, seed=0, verbose=False, device='cuda', resc_schedule=False)
        
        if self.attack_type == 'PGD':
            self.attacks_to_run = [self.pgd]

        elif self.attack_type == 'MTPGD':
            self.attacks_to_run = [self.mtpgd]
      
        elif self.attack_type == 'SFMT':
            self.attacks_to_run = [self.sfmt]
      
        elif self.attack_type == 'APGD':
            self.attacks_to_run = [self.apgd]
            
        elif self.attack_type == 'APGDT':
            self.attacks_to_run = [self.apgdt]
            
        elif self.attack_type == 'MD':
            self.attacks_to_run = [self.md]
            
        elif self.attack_type == 'SFM':
            self.attacks_to_run = [self.sfm]
        
        elif self.attack_type == 'MDMT':
            self.attacks_to_run = [self.mdmt]
        
        elif self.attack_type == 'FAB':
            self.attacks_to_run = [self.fab]
        
        elif self.attack_type == 'Square':
            self.attacks_to_run = [self.square]
        
        elif self.attack_type == 'MD+SFM':
            self.attacks_to_run = [self.md,self.sft]
            
        elif self.attack_type == 'SFM+Squre':
            self.attacks_to_run = [self.sfm,self.square]
        
        elif self.attack_type == 'SFM+MIFPE':
            self.pgd = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=100,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="MIFPE")
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.pgd,self.sfm]
        
        elif self.attack_type == 'MIFPE+MIFPE_T':
            self.pgd1 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="MIFPE")
            self.pgd2 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="MIFPE_T")
            self.attacks_to_run = [self.pgd1,self.pgd2]
            
        elif self.attack_type == 'PGD_SFM+MAR':
            self.pgd1 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="Margin")
            self.pgd2 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="SFM")
            self.attacks_to_run = [self.pgd1,self.pgd2]
        
        elif self.attack_type == 'PGD_SFM+MAR+Dlr':
            self.pgd1 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="Margin")
            self.pgd2 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="SFM")
            self.pgd3 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="Dlr")
            self.attacks_to_run = [self.pgd1,self.pgd2,self.pgd3]
        
        elif self.attack_type == 'PGD_Dlr+MAR':
            self.pgd1 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="Margin")
            self.pgd2 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="Dlr")
            self.attacks_to_run = [self.pgd1,self.pgd2]
        
        elif self.attack_type == 'PGD_CE+Dlr+MAR':
            self.pgd1 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="Margin")
            self.pgd2 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="Dlr")
            self.pgd3 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="CE")
            self.attacks_to_run = [self.pgd1,self.pgd2,self.pgd3]
        
        elif self.attack_type == 'PGD_MIFPE+Dlr+MAR':
            self.pgd1 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="Margin")
            self.pgd2 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="Dlr")
            self.pgd3 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="MIFPE")
            self.attacks_to_run = [self.pgd1,self.pgd2,self.pgd3]
        
        elif self.attack_type == 'PGD_MIFPE+CE':
            self.pgd1 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="MIFPE")
            self.pgd2 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="CE")
            self.attacks_to_run = [self.pgd1,self.pgd2]
    
        elif self.attack_type == 'PGD_CE_T+CE':
            self.pgd1 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="CE_T")
            self.pgd2 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="CE")
            self.attacks_to_run = [self.pgd1,self.pgd2] 
    
        elif self.attack_type == '100MD+100PGD_Mar':
            self.pgd = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=100,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="CE_T")
            self.md = MD.MDAttack(self.model,num_classes=self.num_classes,num_steps=100,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.pgd,self.md]
            
        elif self.attack_type == 'MD+MDMT':
            self.attacks_to_run = [self.md,self.mdmt]
            
        elif self.attack_type == 'APGD_CE+APGDT':
            self.apgd = autopgd_pt.APGDAttack(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss='CE', eot_iter=1, rho=.75, verbose=False)
        
            self.apgdt = autopgd_pt.APGDAttack_targeted(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                    seed=0, eot_iter=1, rho=.75, verbose=False)
            self.attacks_to_run = [self.apgd,self.apgdt]
        
        elif self.attack_type == 'APGD_SFM+APGDT':
            self.apgd = autopgd_pt.APGDAttack(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss='SFM', eot_iter=1, rho=.75, verbose=False)
            self.apgdt = autopgd_pt.APGDAttack_targeted(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                    seed=0, eot_iter=1, rho=.75, verbose=False)
            self.attacks_to_run = [self.apgd,self.apgdt]
        
        elif self.attack_type == 'APGD_CE+MIFPE':
            self.apgd1 = autopgd_pt.APGDAttack(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss='CE', eot_iter=1, rho=.75, verbose=False)
            self.apgd2 = autopgd_pt.APGDAttack(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss='MIFPE', eot_iter=1, rho=.75, verbose=False)
            self.attacks_to_run = [self.apgd1,self.apgd2]
        
        elif self.attack_type == 'APGD_MIFPE+APGDT':
            self.apgd = autopgd_pt.APGDAttack(self.model, n_iter=100, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss='MIFPE', eot_iter=1, rho=.75, verbose=False)
            self.apgdt = autopgd_pt.APGDAttack_targeted(self.model, n_iter=100, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, eot_iter=1, rho=.75, verbose=False)
            self.attacks_to_run = [self.apgd,self.apgdt]
        
        elif self.attack_type == 'APGD_CE+SFM':
            self.apgd1 = autopgd_pt.APGDAttack(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss='CE', eot_iter=1, rho=.75, verbose=False)
            self.apgd2 = autopgd_pt.APGDAttack(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss='SFM', eot_iter=1, rho=.75, verbose=False)
            self.attacks_to_run = [self.apgd1,self.apgd2]
        #SFM+
        elif self.attack_type == 'SFM+APGDT':
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.apgdt = autopgd_pt.APGDAttack_targeted(self.model, n_iter=100, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, eot_iter=1, rho=.75, verbose=False)
            self.attacks_to_run = [self.sfm,self.apgdt]

        elif self.attack_type == 'SFM+PGD_Dlr':
            self.pgd = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=100,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="Dlr")
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.pgd,self.sfm]
            
        elif self.attack_type == 'SFM+PGD_MIFPE':
            self.pgd = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=100,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="MIFPE")
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.pgd,self.sfm]
        
        elif self.attack_type == '100SFM+100PGD_Mar':
            self.pgd = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=100,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="CE_T")
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=100,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.pgd,self.sfm]
        
        elif self.attack_type == '100SFM+100PGD_CE':
            self.pgd = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=100,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="CE")
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=100,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.pgd,self.sfm]
            
        elif self.attack_type == '100SFM+100PGD_CE+CE_T':
            self.pgd1 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=100,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="CE")
            self.pgd2 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=100,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="CE_T")
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=100,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.pgd1,self.pgd2,self.sfm]
        
        elif self.attack_type == 'SFM+PGD_AltPGD':
            self.pgd = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=100,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type="AltPGD")
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.pgd,self.sfm]
                
        elif self.attack_type == 'SFM+APGD_CE':
            self.apgd = autopgd_pt.APGDAttack(self.model, n_iter=100, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss='CE', eot_iter=1, rho=.75, verbose=False)
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.apgd,self.sfm]
            
        elif self.attack_type == 'SFM+APGD_Dlr':
            self.apgd = autopgd_pt.APGDAttack(self.model, n_iter=100, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss='Dlr', eot_iter=1, rho=.75, verbose=False)
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.apgd,self.sfm]
            
        elif self.attack_type == 'SFM+APGD_Margin':
            self.apgd = autopgd_pt.APGDAttack(self.model, n_iter=100, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss='Margin', eot_iter=1, rho=.75, verbose=False)
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.apgd,self.sfm]
        
        elif self.attack_type == 'SFM+APGD_SFM':
            self.apgd = autopgd_pt.APGDAttack(self.model, n_iter=100, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss='SFM', eot_iter=1, rho=.75, verbose=False)
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.apgd,self.sfm]
            
        elif self.attack_type == 'SFM+APGD_MIFPE':
            self.apgd = autopgd_pt.APGDAttack(self.model, n_iter=100, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss='MIFPE', eot_iter=1, rho=.75, verbose=False)
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.attacks_to_run = [self.apgd,self.sfm]
        
        elif self.attack_type == 'SFMT+APGDT':
            self.sfmt = MD.SFMTAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.apgdt = autopgd_pt.APGDAttack_targeted(self.model, n_iter=100, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, eot_iter=1, rho=.75, verbose=False)
            self.attacks_to_run = [self.sfmt,self.apgdt]
        
        elif self.attack_type == 'SFM+SFMT':
            self.attacks_to_run = [self.sfmt,self.sfmt]
        
        elif self.attack_type == 'SFM+SFMT+FAB+Square':
            self.sfm = MD.SFMAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.sfmt = MD.SFMTAttack(self.model,num_classes=self.num_classes,num_steps=50,num_random_starts=self.num_restarts,epsilon=self.eps)
            self.fab = fab_pt.FABAttack_PT(model, n_restarts=self.num_restarts, n_iter=self.num_steps,
                                       eps=self.eps, seed=0,n_target_classes=9,
                                       verbose=False, device='cuda')
            self.square = square.SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.eps, norm='Linf',
                n_restarts=1, seed=0, verbose=False, device='cuda', resc_schedule=False)
            self.attacks_to_run = [self.sfm,self.sfmt,self.fab,self.square]
            
            
    def evaluate(self):
        clean_count = 0
        adv_count = 0
        total = 0
        adv_index = []
        for images, labels in self.data_loader:
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
            
            adv_i = (adv_pred != labels) #
            adv_index.extend(adv_i.tolist()) #
            
            adv_correct = (adv_pred == labels).sum().item()
            adv_count += adv_correct

            # Log
            if self.verbose:
                rs = (clean_count, total, clean_count * 100 / total,
                      adv_count, total, adv_count * 100 / total)
                payload = (('Clean: %d/%d Clean Acc: %.2f Adv: %d/%d '+ 'Adv_Acc: %.2f') % rs)
                print(payload)

        adv_indexs = [str(i) for i in range(len(adv_index)) if adv_index[i]]

        clean_acc = clean_count * 100 / total
        adv_acc = adv_count * 100 / total
        return clean_acc, adv_acc,adv_indexs
    
    '''
    def evaluate(self):
        nc = 0
        sum = 0
        adv_nc = 0

        for images, labels in tqdm(self.data_loader):
            # Build x_adv
            X_adv = images.clone().to(device)
            images = images.to(device)
            labels = labels.to(device)
            images.requires_grad_()
            
            # Check Clean Acc
            with torch.no_grad():
                clean_logits = self.model(images)

            clean_pred = clean_logits.data.max(1)[1].detach()
            acc = clean_pred == labels
            cur_c = acc.squeeze()
            nc += cur_c.sum()
            sum += len(cur_c)

            # All attacks and update x_adv
            # 以上一个攻击作为初始化
            if self.continue_attack:
                for a in self.attacks_to_run:
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape)==0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_fool,y_fool = images[ind_to_fool], labels[ind_to_fool]
                        x_p = a.perturb(x_fool, y_fool).detach()
                        
                        with torch.no_grad():
                            logits = self.model(x_p)
                            
                        x_pred = logits.data.max(1)[1].detach()
                        acc_curr = (x_pred == y_fool)
                        ind_curr = (acc_curr==0).nonzero().squeeze()
                        acc[ind_to_fool[ind_curr]] = 0 
                        images.requires_grad_(False)
                        images[ind_to_fool[ind_curr]] = x_p[ind_curr]
                        images.requires_grad_()
            # 不以上一个攻击作为初始化
            else:
                for a in self.attacks_to_run:
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape)==0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_fool,y_fool = images[ind_to_fool], labels[ind_to_fool]
                        x_p = a.perturb(x_fool, y_fool).detach()
                        
                        with torch.no_grad():
                            logits = self.model(x_p)
                        x_pred = logits.data.max(1)[1].detach()
                        acc_curr = (x_pred == y_fool)
                        ind_curr = (acc_curr==0).nonzero().squeeze()
                        acc[ind_to_fool[ind_curr]] = 0 
                        X_adv[ind_to_fool[ind_curr]] = x_p[ind_curr].clone()
            if self.continue_attack:
                X_adv = images.clone()
            with torch.no_grad():
                adv_pre = self.model(X_adv).max(1)[1].detach()
            adv_a = (adv_pre == labels).squeeze()
            adv_nc += adv_a.sum()

            # Log
            if self.verbose:
                print(f"\033[1;31;40m {self.attack_type} - clean_acc: {cur_c.sum()/len(cur_c)*100:.2f}% - adv_acc: {adv_a.sum()/len(adv_a)*100:.2f}% \033[0m")
            
        clean_acc = nc/sum
        adv_acc = adv_nc/sum
        print(adv_acc)
        return clean_acc.item(), adv_acc.item()
        '''
