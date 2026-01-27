import pandas as pd
import torch
import numpy as np
from tqdm import tqdm


def compute_validation_map(model, val_embeddings, val_labels, device):

    model.eval()

    with torch.no_grad():
        val_tensor = torch.FloatTensor(val_embeddings).to(device)
        finetuned_emb = model(val_tensor)

    sim_matrix = torch.cosine_similarity(finetuned_emb, finetuned_emb).cpu().numpy()
    sim_matrix = torch.matmul(finetuned_emb, finetuned_emb.T).cpu().numpy()
    sim_matrix /= np.linalg.norm(finetuned_emb.cpu().numpy(), axis=1, keepdims=True)
    sim_matrix /= np.linalg.norm(finetuned_emb.cpu().numpy(), axis=1, keepdims=True).T

    np.fill_diagonal(sim_matrix, -1)
    query_aps = {}

    for query_idx in range(len(val_labels)):
        query_label = val_labels[query_idx]
        similarities = sim_matrix[query_idx]

        gallery_labels = val_labels.copy()
        is_match = (gallery_labels == query_label).astype(int)
        is_match[query_idx] = 0  # Exclude self

        sorted_indices = np.argsort(-similarities)
        sorted_matches = is_match[sorted_indices]

        n_positives = sorted_matches.sum()
        if n_positives == 0:
            continue

        cumsum = np.cumsum(sorted_matches)
        precision_at_k = cumsum / np.arange(1, len(sorted_matches) + 1)
        ap = np.sum(precision_at_k * sorted_matches) / n_positives

        query_aps[query_idx] = (query_label, ap)

    identity_aps = {}
    for query_idx, (label, ap) in query_aps.items():
        if label not in identity_aps:
            identity_aps[label] = []
        identity_aps[label].append(ap)

    identity_mean_aps = [np.mean(aps) for aps in identity_aps.values()]
    balanced_map = np.mean(identity_mean_aps)

    return balanced_map


def build_submission(submission_path, model, test_query_dataset, test_gallery_dataset, device):
    model.eval()

    similarities = []
    test_query_embeddings = test_query_dataset.get_embeddings()
    test_gallery_embeddings = test_gallery_dataset.get_embeddings()
    with torch.no_grad():
        test_query_embeddings = torch.FloatTensor(test_query_embeddings).to(device)
        test_query_embeddings = model(test_query_embeddings).cpu().numpy()
        test_gallery_embeddings = torch.FloatTensor(test_gallery_embeddings).to(device)
        test_gallery_embeddings = model(test_gallery_embeddings).cpu().numpy()

    for test_query_idx, test_gallery_idx in tqdm(
        zip(test_query_dataset.data.index, test_gallery_dataset.data.index), total=len(test_query_dataset.data), desc="Building submission"
    ):
        query_emb = test_query_embeddings[test_query_idx]
        gallery_emb = test_gallery_embeddings[test_gallery_idx]
        sim = np.dot(query_emb, gallery_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(gallery_emb))
        similarities.append(sim)

    # Clip to [0, 1] range
    similarities = np.array(similarities)
    similarities = np.clip(similarities, 0.0, 1.0)
    submission_df = pd.DataFrame({"row_id": list(range(len(similarities))), "similarity": similarities})
    if not submission_path.parent.exists():
        submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    return submission_path
