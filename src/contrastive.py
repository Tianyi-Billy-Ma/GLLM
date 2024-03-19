
import torch
import torch.nn.functional as F

def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor, T):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def contrastive_loss_node(x1, x2, T):
    
    l1 = semi_loss(x1, x2, T)
    l2 = semi_loss(x2, x1, T)

    ret = (l1 + l2) * 0.5
    ret = ret.mean()
    
    return ret