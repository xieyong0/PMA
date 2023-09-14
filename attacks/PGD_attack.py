import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import adv_check_and_update, one_hot_tensor

def Margin_loss(logits, y):
    logit_org = logits.gather(1, y.view(-1, 1))
    logit_target = logits.gather(1, (logits - torch.eye(10)[y.to("cpu")].to("cuda") * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss

def Dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
        1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

def Dlr_loss_targeted(self, x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x[u, self.y_target]) / (x_sorted[:, -1] - .5 * (
        x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

class PGD_attack():
    def __init__(self, model, eps=8./255., random_start=False,
                 noise = "Uniform", num_restarts=1, step_size=2./255,
                 num_steps=50, loss_f='CE', use_odi=False, opt='SGD'):
        self.model = model
        self.eps = eps
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start
        self.num_restarts = num_restarts
        self.use_odi = use_odi
        self.noise = noise
        self.loss_f = loss_f
        self.opt = opt
    
    def perturb(self, x_in, y_in):
        model = self.model
        device = x_in.device
        eps = self.eps
        X_adv = x_in.detach().clone()
        X_pgd = Variable(x_in.data, requires_grad=True)
        nc = torch.zeros_like(y_in)

        for r in range(self.num_restarts):
            if self.random_start:
                if self.noise == 'Uniform':
                    random_noise = torch.FloatTensor(*x_in.shape).uniform_(-eps,eps).to(device)
                elif self.noise == 'Gaussian':
                    random_noise = torch.randn(*x_in.shape).to(device)
                X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

            if self.use_odi:
                out = model(x_in)
                rv = torch.FloatTensor(*out.shape).uniform_(-1., 1.).to(device)

            for i in range(self.num_steps):
                if self.use_odi and i < 2:
                    loss = (model(X_pgd) * rv).sum()
                else:
                    if self.loss_f == 'CE':
                        loss = F.cross_entropy(model(X_pgd), y_in)
                    elif self.loss_f == 'Margin':
                        loss = Margin_loss(model(X_pgd), y_in)
                    elif self.loss_f == 'Dlr':
                        loss = Dlr_loss(model(X_pgd), y_in).sum()
                loss.backward()
                eta = self.step_size * X_pgd.grad.data.sign()
                X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
                eta = torch.clamp(X_pgd.data - x_in.data, -eps, eps)
                X_pgd = Variable(x_in.data + eta, requires_grad=True)
                X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
                logits = self.model(X_pgd)
                X_adv, nc = adv_check_and_update(X_pgd, logits, y_in, nc, X_adv)
        return X_adv
                
