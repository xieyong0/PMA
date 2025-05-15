import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

torch.manual_seed(0)

class MDAttack():
    """Momentum Decay Attack for generating adversarial examples.
    
    Attributes:
        model: Target model to attack
        epsilon: Maximum perturbation allowance (L-inf norm)
        num_steps: Number of attack iterations
        step_size: Base step size for gradient updates
        num_random_starts: Number of random restarts
        v_min: Minimum pixel value (typically 0 for images)
        v_max: Maximum pixel value (typically 1 for images)
        change_point: Step number to switch loss strategies
        first_step_size: Initial step size for early iterations
        seed: Random seed for reproducibility
        norm: Norm type for perturbation constraint (L-inf supported)
        num_classes: Number of classes in classification task
        use_odi: Whether to use Online Decay Initialization
        use_dlr: Whether to use DLR loss (not implemented in current code)
        loss_fn: Loss function type ('margin' or 'p_margin')
        initial_step_size: Calculated initial step size based on epsilon
    """
    
    def __init__(self, model, epsilon=8./255., num_steps=50, step_size=2./255.,
                 num_random_starts=1, v_min=0., v_max=1., change_point=50,
                 first_step_size=16./255., seed=0, norm='Linf', num_classes=100,
                 use_odi=False, use_dlr=False, loss_fn='margin'):
        # Initialize attack parameters
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_random_starts = num_random_starts
        self.v_min = v_min
        self.v_max = v_max
        self.change_point = num_steps/2  # Mid-point for strategy change
        self.first_step_size = first_step_size
        self.seed = seed
        self.norm = norm
        self.num_classes = num_classes
        self.use_odi = use_odi
        self.use_dlr = use_dlr
        self.loss_fn = loss_fn
        self.initial_step_size = 2.0 * epsilon  # Calculate initial step size

    def perturb(self, x_in, y_in):
        """Generate adversarial examples for input batch.
        
        Args:
            x_in: Clean input samples (Tensor)
            y_in: True labels (Tensor)
            
        Returns:
            X_adv: Adversarial examples
            accs: Attack success mask (True where attack failed)
        """
        device = x_in.device
        assert self.loss_fn in ['Margin', 'PMargin']
        
        change_point = self.change_point
        X_adv = x_in.detach().clone()

        # Initial forward pass to find correctly classified samples
        with torch.no_grad():
            logits = self.model(x_in)
        pred = logits.max(dim=1)[1]
        accs = pred == y_in  # Mask of correctly classified samples

        # Attack loop with random restarts
        for _ in range(max(self.num_random_starts, 1)):
            for r in range(2):  # Two-phase attack strategy
                # Add initial random noise within epsilon bound
                r_noise = torch.FloatTensor(*x_in[accs].shape).uniform_(-self.epsilon, self.epsilon).to(device)
                X_adv[accs] = x_in[accs].data + r_noise
                cor_indexs = accs.nonzero().squeeze()  # Indices of currently correct samples
                x_pgd = Variable(X_adv[cor_indexs] + 0., requires_grad=True)
                y = y_in[cor_indexs]
                
                # Iterative attack steps
                for i in range(self.num_steps):
                    with torch.enable_grad():
                        logits = self.model(x_pgd)
                        
                        # Calculate loss based on current strategy
                        if self.loss_fn == "PMargin":
                            logits = F.softmax(logits, dim=-1) 
                        z_y = logits.gather(1, y.view(-1, 1))  # Logits for true labels
                        z_max = logits.gather(1, (logits - torch.eye(self.num_classes)[y.cpu()].to(device) * 9999).argmax(1, keepdim=True))  # Max non-true logits

                        # Dynamic loss selection based on attack phase
                        if i < 1:  # Initial phase
                            loss_per_sample = z_y
                        elif i < change_point:  # Middle phase
                            loss_per_sample = z_max if r else -z_y
                        else:  # Final phase
                            loss_per_sample = z_max - z_y
                            
                        loss = torch.mean(loss_per_sample)
                        loss.backward()
                        
                        # Update success mask
                        acc = logits.max(1)[1] == y
                        accs[cor_indexs] = acc
                        X_adv[cor_indexs] = x_pgd.detach()

                    # Adaptive step size scheduling with cosine annealing
                    if self.use_odi and i < 2:
                        alpha = self.epsilon
                    elif i > change_point:
                        alpha = self.initial_step_size * 0.5 * (1 + np.cos((i-change_point - 1) / (self.num_steps-change_point) * np.pi))
                    else:
                        alpha = self.initial_step_size * 0.5 * (1 + np.cos((i - 1) / (self.num_steps-change_point) * np.pi))
                        
                    # Update adversarial examples with sign gradient
                    eta = alpha * x_pgd.grad.data.sign()
                    x_pgd = x_pgd.detach() + eta.detach()
                    # Project back to epsilon neighborhood and valid pixel range
                    x_pgd = torch.min(torch.max(x_pgd, x_in[cor_indexs] - self.epsilon), x_in[cor_indexs] + self.epsilon)
                    x_pgd = torch.clamp(x_pgd, self.v_min, self.v_max)
                    
                    # Prepare for next iteration
                    x_pgd = Variable(x_pgd[acc], requires_grad=True)
                    cor_indexs = accs.nonzero().squeeze()
                    y = y_in[cor_indexs]
                    
                # Final check of attack success
                with torch.no_grad():
                    logits = self.model(x_pgd)
                acc = logits.max(1)[1] == y
                accs[cor_indexs] = acc
                X_adv[cor_indexs] = x_pgd
                    
        return X_adv, accs
