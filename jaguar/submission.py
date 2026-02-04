from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F


def build_submission(submission_path, model, test_dataloader, device, query_expansion_enabled=False, k_reciprocal_reranking_enabled=False, tta_enabled=False):

    model.eval()
    all_embeddings = []
    submission_path = Path(submission_path)
    if not submission_path.parent.exists():
        submission_path.parent.mkdir(parents=True, exist_ok=True)
    if submission_path.exists():
        submission_path.unlink()

    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc="Generating Embeddings", leave=False)
        for batch in pbar:
            images, _ = batch
            images = images.to(device)

            if tta_enabled:
                # Generate embeddings for both original and horizontally flipped images
                embeddings_original = model(images)
                embeddings_original = F.normalize(embeddings_original, p=2, dim=1)

                # Flip images horizontally
                images_flipped = torch.flip(images, dims=[3])  # Flip along width dimension
                embeddings_flipped = model(images_flipped)
                embeddings_flipped = F.normalize(embeddings_flipped, p=2, dim=1)

                # Average the embeddings
                embeddings = (embeddings_original + embeddings_flipped) / 2.0
                embeddings = F.normalize(embeddings, p=2, dim=1)
            else:
                embeddings = model(images)
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

    if query_expansion_enabled:
        all_embeddings = query_expansion(all_embeddings.cpu().numpy(), k=3)

    sim_matrix = torch.mm(all_embeddings, all_embeddings.t()).cpu().numpy()

    if k_reciprocal_reranking_enabled:
        sim_matrix = k_reciprocal_reranking(all_embeddings.cpu().numpy(), k1=20, k2=6, lambda_value=0.3)

    sim_matrix = np.clip(sim_matrix, a_min=0.0, a_max=1.0)

    submission_rows = []
    num_samples = all_embeddings.shape[0]
    c = 0
    for i in range(num_samples):
        similarities = sim_matrix[i]
        for j in range(num_samples):
            if i == j:
                continue
            row_id = c
            c += 1
            similarity = similarities[j]
            submission_rows.append({"row_id": row_id, "similarity": similarity})
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(submission_path, index=False)


def query_expansion(emb, k=5):
    """Perform query expansion on the given embeddings.

    Args:
        emb (np.ndarray): The input embeddings of shape (N, D).
        k (int): The number of nearest neighbors to consider for expansion.

    Returns:
        torch.Tensor: The expanded embeddings of shape (N, D).
    """
    emb = torch.from_numpy(emb)
    sim_matrix = torch.mm(emb, emb.T)
    sim_matrix.fill_diagonal_(-1)

    expanded_emb = []
    for i in range(emb.size(0)):
        similarities = sim_matrix[i]
        topk_indices = torch.topk(similarities, k=k, largest=True).indices
        neighbor_embs = emb[topk_indices]
        expanded_vector = torch.mean(torch.cat([emb[i].unsqueeze(0), neighbor_embs], dim=0), dim=0)
        expanded_vector = F.normalize(expanded_vector, p=2, dim=0)
        expanded_emb.append(expanded_vector)

    expanded_emb = torch.stack(expanded_emb, dim=0)
    return expanded_emb


def k_reciprocal_reranking(embeddings, k1=20, k2=6, lambda_value=0.3):
    """Perform k-reciprocal re-ranking on the given embeddings.

    Args:
        embeddings (np.ndarray): The input embeddings of shape (N, D).
        k1 (int): The number of nearest neighbors for initial ranking.
        k2 (int): The number of nearest neighbors for expansion.
        lambda_value (float): The weight for combining original and Jaccard distances.

    Returns:
        np.ndarray: The re-ranked similarity matrix of shape (N, N).
    """

    # Compute original distance
    original_dist = np.matmul(embeddings, embeddings.T)
    original_dist = 1 - original_dist
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))

    # Initialize variables
    num_samples = embeddings.shape[0]
    V = np.zeros_like(original_dist).astype(np.float32)

    # k-reciprocal neighbors
    for i in range(num_samples):
        forward_k_neigh_index = np.argsort(original_dist[i])[: k1 + 1]
        backward_k_neigh_index = np.array([np.argsort(original_dist[j])[: k1 + 1] for j in forward_k_neigh_index])
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]

        # Expansion
        for j in k_reciprocal_index:
            candidate_forward_k_neigh_index = np.argsort(original_dist[j])[: int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = np.array([np.argsort(original_dist[m])[: int(np.around(k1 / 2)) + 1] for m in candidate_forward_k_neigh_index])
            fi_candidate = np.where(candidate_backward_k_neigh_index == j)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_index = np.append(k_reciprocal_index, candidate_k_reciprocal_index)

        k_reciprocal_index = np.unique(k_reciprocal_index)
        weight = np.exp(-original_dist[i][k_reciprocal_index])
        V[i][k_reciprocal_index] = weight / np.sum(weight)
    # Jaccard distance
    original_dist = original_dist[:num_samples, :]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(num_samples):
            V_qe[i, :] = np.mean(V[np.argsort(original_dist[i])[:k2], :], axis=0)
        V = V_qe
        del V_qe
    invIndex = []
    for i in range(num_samples):
        invIndex.append(np.where(V[:, i] != 0)[0])
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(num_samples):
        temp_min = np.zeros((1, num_samples), dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        for j in indNonZero:
            temp_min[0, invIndex[j]] += np.minimum(V[i, j], V[invIndex[j], j])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
    # Final distance
    final_dist = (1 - lambda_value) * jaccard_dist + lambda_value * original_dist
    final_dist = final_dist[:num_samples,]
    final_sim = 1 - final_dist
    return final_sim
