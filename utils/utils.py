import torch
import torch.nn.functional as F


def gram_schmidt_process(target_grad, other_grads, method="orthonormal_basis", normalize=False):
    """
    Apply Gram-Schmidt process to get orthogonal gradient
    
    Args:
        target_grad: gradient vector to be projected (forget gradient)
        other_grads: list of other gradient vectors [remain gradient, fair gradient]
        method: "sequential_projection" - subtract projections against raw gradients sequentially
                "orthonormal_basis" - first build orthonormal basis for the subspace, then project
        normalize: if True, normalize the result to unit norm
    """
    def flatten_grad(grad):
        return torch.cat([g.flatten() for g in grad])
    
    def reshape_grad(grad_vector, template_grad):
        grads = []
        start_idx = 0
        for template in template_grad:
            size = template.numel()
            grad_piece = grad_vector[start_idx:start_idx + size].reshape(template.shape)
            grads.append(grad_piece)
            start_idx += size
        return grads

    v = flatten_grad(target_grad)
    U = [flatten_grad(g) for g in other_grads]

    if method == "sequential_projection":
        for u in U:
            norm_sq = torch.dot(u, u)
            if norm_sq > 1e-16:
                proj_coef = torch.dot(v, u) / norm_sq
                v = v - proj_coef * u

    elif method == "orthonormal_basis":
        basis = []
        for u in U:
            q = u.clone()
            for b in basis:
                q = q - torch.dot(q, b) * b
            norm_q = torch.norm(q)
            if norm_q > 1e-16:
                basis.append(q / norm_q)
        for b in basis:
            v = v - torch.dot(v, b) * b
    else:
        raise ValueError(f"Unknown gram_schmidt method: {method}")

    if normalize:
        norm_v = torch.norm(v)
        if norm_v > 1e-16:
            v = v / norm_v

    return reshape_grad(v, target_grad)


def flatten_grads(grad_list):
    """Helper function to flatten gradients for similarity computation"""
    return torch.cat([g.flatten() for g in grad_list if g is not None])


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two gradient vectors"""
    if vec1.norm() < 1e-8 or vec2.norm() < 1e-8:
        return 0.0
    return torch.dot(vec1, vec2) / (vec1.norm() * vec2.norm())
