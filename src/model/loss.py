import torch
import torch.nn.functional as F

@torch.no_grad()
def sample_b_indices(edge_index: torch.Tensor, B: int) -> torch.Tensor:
    num_edges = edge_index.size(1)
    if B > num_edges:
        B = num_edges
    perm = torch.randperm(num_edges)[:B]
    return perm

def contrastive_loss(
        z1: torch.Tensor, 
        z2: torch.Tensor, 
        temperature: float = 0.5
    ) -> torch.Tensor:
    
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T) / temperature

    labels = torch.arange(batch_size).to(z1.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(similarity_matrix, labels)

    return loss