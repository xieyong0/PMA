"""
PyTorch implementation of Projected Gradient Descent (PGD) adversarial attack.
Supports multiple loss functions and adaptive step size scheduling.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# Set random seed for reproducibility
torch.manual_seed(0)

def Dlr_loss(x, y):
    """
    Difference of Logits Ratio (DLR) loss from Croce et al. 2020
    Args:
        x: model outputs (logits) with shape [batch_size, num_classes]
        y: ground truth labels with shape [batch_size]
    Returns:
        DLR loss value for each sample in batch
    """
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()  # Indicator for correct class being top-1
    batch_range = torch.arange(x.shape[0])
    return -(x[batch_range, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
        1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

def Dlr_loss_t(x, y, y_target):
    """
    Targeted version of DLR loss
    Args:
        x: model outputs (logits)
        y: original ground truth labels
        y_target: target attack labels 
    """
    x_sorted, ind_sorted = x.sort(dim=1)
    return -(x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]) / (
        x_sorted[:, -1] - 0.5 * x_sorted[:, -3] - 0.5 * x_sorted[:, -4] + 1e-12)

def Margin_loss(logits, y, num_classes):
    """
    Margin loss: difference between true class logit and max non-true class logit
    Args:
        logits: model outputs [batch_size, num_classes]
        y: true labels [batch_size]
        num_classes: number of classes
    """
    device = logits.device
    logit_org = logits.gather(1, y.view(-1, 1))
    # Get max logit excluding true class
    logit_target = logits.gather(1, (logits - torch.eye(num_classes)[y.to("cpu")].to(device) * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    return torch.sum(loss)

def P_Margin(logits, y, num_classes):
    """
    Margin loss applied on softmax probabilities instead of logits
    """
    device = logits.device
    logits = F.softmax(logits, dim=-1)
    logit_org = logits.gather(1, y.view(-1, 1))
    logit_target = logits.gather(1, (logits - torch.eye(num_classes)[y.to("cpu")].to(device) * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    return torch.sum(loss)
    
def MIFPE(logits, y):
    """
    Maximum Interval Probability with Temperature Scaling (MIFPE) loss
    Dynamically adjusts logits scale based on output confidence
    """
    t = 1.0  # Temperature parameter
    device = logits.device
    
    def get_output_scale(output):
        """Calculate per-sample scaling factor based on top-2 logit differences"""
        std_max_out = []
        maxk = max((10,))
        pred_val_out, _ = output.topk(maxk, 1, True, True)
        std_max_out.extend((pred_val_out[:, 0] - pred_val_out[:, 1]).cpu().numpy())
        scale_list = [item / t for item in std_max_out]
        return torch.tensor(scale_list).to(device).unsqueeze(-1)

    scale_output = get_output_scale(logits.clone().detach())
    scaled_logits = logits / scale_output
    return F.cross_entropy(scaled_logits, y)

class PGDAttack:
    """
    Projected Gradient Descent (PGD) adversarial attacker
    Features:
    - Multiple loss functions support (CE, DLR, Margin, etc.)
    - Step size scheduling (constant/linear/cosine decay)
    - Random restarts
    - ODI initialization option
    """
    
    def __init__(self, model, epsilon=8./255., num_steps=50, step_size=2./255.,
                 num_restarts=1, v_min=0., v_max=1., num_classes=10,
                 random_start=False, loss_type='CE', decay_step='cos', use_odi=False):
        """
        Initialize PGD attacker
        Args:
            model: victim model to attack
            epsilon: maximum perturbation bound (L-infinity norm)
            num_steps: number of attack iterations
            step_size: base attack step size
            num_restarts: number of random restarts
            v_min: minimum pixel value (0 for image normalization)
            v_max: maximum pixel value (1 for image normalization)
            num_classes: number of classification classes
            random_start: whether to use random initialization
            loss_type: attack loss type ('CE', 'Dlr', 'Margin', etc.)
            decay_step: step size schedule ('cos' or 'linear')
            use_odi: enable Orthogonal Direction Iteration initialization
        """
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.random_start = random_start
        self.num_restarts = num_restarts
        self.v_min = v_min
        self.v_max = v_max
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.decay_step = decay_step
        self.use_odi = use_odi

    def _get_rand_noise(self, X):
        """Generate uniform random noise within [-epsilon, epsilon]"""
        return torch.FloatTensor(*X.shape).uniform_(-self.epsilon, self.epsilon).to(X.device)

    def perturb(self, x_in, y_in):
        """
        Generate adversarial examples
        Args:
            x_in: clean input images [batch_size, ...]
            y_in: true labels [batch_size]
        Returns:
            adv_examples: generated adversarial examples
            success_mask: boolean mask indicating successful attacks
        """
        model = self.model
        device = x_in.device
        X_adv = x_in.detach().clone()
        step_size_begin = 2.0/255  # Initial step size

        # Initial clean evaluation
        with torch.no_grad():
            clean_logits = model(x_in)
            if isinstance(clean_logits, list):  # Handle multi-output models
                clean_logits = clean_logits[-1]
            clean_pred = clean_logits.data.max(1)[1].detach()
            accs = clean_pred == y_in  # Initial accuracy mask

        # Multiple restarts for improved attack success rate
        for _ in range(self.num_restarts):
            if self.random_start:
                random_noise = self._get_rand_noise(x_in[accs])
                X_pgd = torch.clamp(x_in[accs] + random_noise, self.v_min, self.v_max)
            else:
                X_pgd = x_in[accs]

            # ODI initialization setup
            if self.use_odi:
                out = model(x_in)
                rv = torch.FloatTensor(*out.shape).uniform_(-1., 1.).to(device)

            cor_indexs = accs.nonzero().squeeze()  # Indices of currently correct predictions
            x_pgd = Variable(X_pgd[cor_indexs], requires_grad=True)
            y = y_in[cor_indexs]

            # Main attack loop
            for i in range(self.num_steps):
                # Adaptive step size scheduling
                if self.decay_step == 'linear':
                    step_size = step_size_begin * (1 - i / self.num_steps)
                elif self.decay_step == 'cos':
                    step_size = step_size_begin * math.cos(i / self.num_steps * math.pi * 0.5)
                
                # Forward pass and loss calculation
                logit = model(x_pgd)
                
                # Loss selection logic
                if self.use_odi and i < 2:  # ODI initialization for first 2 steps
                    loss = (logit * rv).sum()
                elif self.loss_type == 'CE':
                    loss = F.cross_entropy(logit, y)
                elif self.loss_type == 'CE_T':
                    logit_y = torch.eye(self.num_classes)[y.to("cpu")].to("cuda")*logit*1e8
                    target = (logit - logit_y).argmax(dim=-1)
                    loss = -F.cross_entropy(logit, target)
                elif self.loss_type == 'Dlr':
                    loss = Dlr_loss(logit, y).sum()
                elif self.loss_type == 'Margin':
                    loss = Margin_loss(logit, y, self.num_classes)
                elif self.loss_type == 'PMargin':
                    loss = P_Margin(logit, y, self.num_classes)
                elif self.loss_type == 'MIFPE':
                    loss = MIFPE(logit, y)
                elif self.loss_type == 'MIFPE_T':
                    logit_y = torch.eye(self.num_classes)[y.to("cpu")].to("cuda")*logit*1e8
                    target = (logit - logit_y).argmax(dim=-1)
                    loss = -MIFPE(logit, target)
                else:
                    raise ValueError(f"Unsupported loss type: {self.loss_type}")
                
                # Backward pass and gradient update
                loss.backward()
                acc = logit.max(1)[1].detach() == y
                accs[cor_indexs] = acc  # Update success mask
                X_adv[cor_indexs] = x_pgd.detach()  # Update successful adversarial examples

                # Gradient sign step
                if self.use_odi and i < 2:
                    eta = self.epsilon * x_pgd.grad.data.sign()
                else:
                    eta = step_size * x_pgd.grad.data.sign()

                # Projection to epsilon ball and valid pixel range
                x_pgd = torch.clamp(x_pgd.data + eta, self.v_min, self.v_max)
                eta = torch.clamp(x_pgd - x_in[cor_indexs].data, -self.epsilon, self.epsilon)
                x_pgd = torch.clamp(x_in[cor_indexs].data + eta, self.v_min, self.v_max)
                
                # Prepare for next iteration
                x_pgd = Variable(x_pgd[acc], requires_grad=True)
                cor_indexs = accs.nonzero().squeeze()
                y = y_in[cor_indexs]

            # Final update after attack loop
            with torch.no_grad():
                logits = model(x_pgd)
                acc = logits.max(1)[1] == y
                accs[cor_indexs] = acc
                X_adv[cor_indexs] = x_pgd

        return X_adv, accs
