"""
This module implements an adversarial attacker class to evaluate model robustness against various attacks.
Includes PGD, APGD, MD, PMA attacks and their variants.
"""

import torch
import numpy as np
from . import PGD
from . import autopgd_pt
from . import MD
from tqdm.auto import tqdm
import time

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attacker():
    """
    Adversarial attacker class to generate perturbations and evaluate model robustness.
    
    Attributes:
        model (torch.nn.Module): Target model to attack
        attack_type (str): Type of attack to perform (PGD, APGD, MD, etc.)
        eps (float): Maximum perturbation magnitude (L-infinity norm)
        num_steps (int): Number of attack iterations
        step_size (float): Step size per iteration
        num_restarts (int): Number of random restarts for attack
        num_classes (int): Number of classes in classification task
        bs (int): Batch size for evaluation
        logger: Optional logger for recording results
    """
    
    def __init__(self, model, attack_type='PGD_attack', eps=8./255, random_start=False,
                 noise='Uniform', num_restarts=1, step_size=2./255, bs=32,
                 num_steps=100, loss_f='CE', use_odi=False, num_classes=10,
                 verbose=True, x_test=None, y_test=None, logger=None):
        # Initialize attack parameters
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
        self.x_test = x_test
        self.y_test = y_test
        self.bs = bs
        self.logger = logger
        self.verbose = verbose

        # Initialize attack methods
        self._init_attack_methods()

    def _init_attack_methods(self):
        """Initialize attack objects based on attack type."""
        # Standard PGD attack
        self.pgd = PGD.PGDAttack(
            self.model, epsilon=self.eps, num_steps=self.num_steps,
            step_size=self.step_size, num_restarts=self.num_restarts,
            num_classes=self.num_classes, random_start=self.random_start,
            loss_type=self.loss_f
        )

        # Auto-PGD attack (untargeted)
        self.apgd = autopgd_pt.APGDAttack(
            self.model, n_iter=self.num_steps, norm='Linf', 
            n_restarts=self.num_restarts, eps=self.eps, seed=0, 
            loss=self.loss_f, eot_iter=1, rho=.75, verbose=False
        )

        # Margin Decomposition attack
        self.md = MD.MDAttack(
            self.model, num_classes=self.num_classes,
            num_steps=self.num_steps, num_random_starts=self.num_restarts,
            epsilon=self.eps, loss_fn='margin'
        )

        # Probability Masking Attack
        self.pma = MD.MDAttack(
            self.model, num_classes=self.num_classes,
            num_steps=self.num_steps, num_random_starts=self.num_restarts,
            epsilon=self.eps, loss_fn='p_margin'
        )

        # Targeted Auto-PGD attack
        self.apgdt = autopgd_pt.APGDAttack_targeted(
            self.model, n_iter=self.num_steps, norm='Linf', 
            n_restarts=self.num_restarts, eps=self.eps, seed=0, 
            eot_iter=1, rho=.75, verbose=False, loss=self.loss_f
        )

        # Configure attack sequence based on attack type
        self._configure_attack_sequence()

    def _configure_attack_sequence(self):
        """Configure attack sequence based on specified attack type."""
        if self.attack_type == 'PGD':
            self.attacks_to_run = [self.pgd]
        elif self.attack_type == 'APGD':
            self.attacks_to_run = [self.apgd]
        elif self.attack_type == 'APGDT':
            self.attacks_to_run = [self.apgdt]
        elif self.attack_type == 'MD':
            self.attacks_to_run = [self.md]
        elif self.attack_type == 'PMA':
            self.attacks_to_run = [self.pma]
        elif self.attack_type == 'PMA+':
            # Enhanced PMA+ attack configuration
            self.pma = MD.MDAttack(
                self.model, num_classes=self.num_classes,
                num_steps=self.num_steps, num_random_starts=self.num_restarts,
                epsilon=self.eps, loss_fn="PMargin"
            )
            self.apgdt = autopgd_pt.APGDAttack_targeted(
                self.model, n_iter=2, norm='Linf', 
                n_restarts=self.num_restarts, eps=self.eps, seed=0, 
                eot_iter=self.num_steps, rho=.75, verbose=False, loss='Dlr'
            )
            self.attacks_to_run = [self.pma, self.apgdt]
        else:
            raise "method error"

    def evaluate(self):
        """
        Evaluate model robustness against configured attacks.
        
        Returns:
            tuple: (clean accuracy, adversarial accuracy, adversarial examples)
        """
        total = self.y_test.shape[0]
        adv_images = self.x_test.clone()  # Initialize with clean images
        remaining_indices = torch.arange(total)  # Track vulnerable samples
        
        clean_correct = 0
        attack_results = []

        for attack_idx, attack in enumerate(self.attacks_to_run):
            print(f"—————— Attack {attack_idx+1} ——————")
            start_time = time.time()
            vulnerable_indices = torch.tensor([], dtype=torch.int32)

            # Process data in batches
            for batch_start in tqdm(range(0, len(remaining_indices), self.bs)):
                # Get current batch indices
                batch_indices = remaining_indices[batch_start:batch_start+self.bs]
                
                # Get batch data
                images = self.x_test[batch_indices].to(device)
                labels = self.y_test[batch_indices].to(device)

                # Calculate clean accuracy for first attack
                with torch.no_grad():
                    clean_logits = self.model(images)
                    if isinstance(clean_logits, list):  # Handle multi-output models
                        clean_logits = clean_logits[-1]
                    clean_pred = clean_logits.argmax(1)
                    correct_mask = clean_pred == labels

                if attack_idx == 0:  # Only count clean accuracy once
                    clean_correct += correct_mask.sum().item()

                # Generate adversarial examples for correctly classified samples
                correct_indices = correct_mask.nonzero().squeeze()
                if correct_indices.numel() == 0:
                    continue  # Skip if no correct samples

                # Generate adversarial examples
                x_adv, adv_correct = attack.perturb(
                    images[correct_indices], 
                    labels[correct_indices]
                )

                # Update vulnerable samples and adversarial examples
                remaining_mask = adv_correct.cpu()
                vulnerable_indices = torch.cat([
                    vulnerable_indices,
                    batch_indices[correct_indices[remaining_mask].cpu()]
                ])
                
                # Store generated adversarial examples
                adv_images[batch_indices[correct_indices.cpu()]] = x_adv.detach().cpu()

            # Update remaining indices for next attack
            remaining_indices = vulnerable_indices
            attack_acc = len(remaining_indices) / total
            attack_results.append(attack_acc)
            
            # Log attack results
            if self.verbose:
                print(f"Attack {attack_idx+1} Results:")
                print(f"- Time: {time.time()-start_time:.2f}s")
                print(f"- Adversarial Accuracy: {attack_acc*100:.2f}%")

        # Calculate final metrics
        clean_acc = clean_correct / total
        final_adv_acc = len(remaining_indices) / total
        return clean_acc, final_adv_acc, adv_images
