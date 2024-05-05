import torch
import torch.nn.functional as F
import torch.nn as nn


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def semi_loss(z1: torch.Tensor, z2: torch.Tensor, T):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
    )


def contrastive_loss_node(x1, x2, T):
    l1 = semi_loss(x1, x2, T)
    l2 = semi_loss(x2, x1, T)

    ret = (l1 + l2) * 0.5
    ret = ret.mean()

    return ret


class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """

    def __init__(self, temperature=0.5):
        super().__init__()

        self.temperature = temperature
        self.loss_fct = nn.CrossEntropyLoss()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

    def forward(self, proj_1, proj_2, tau=None):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """

        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        cos_sim = torch.einsum("id,jd->ij", z_i, z_j) / self.temperature
        labels = torch.arange(cos_sim.size(0)).long().to(proj_1.device)
        loss = self.loss_fct(cos_sim, labels)

        return loss
