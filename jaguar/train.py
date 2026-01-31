from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_epoch(model, data_loader, optimizer, device) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        data_loader: DataLoader providing the training data.
        optimizer: The optimizer to use for training.
        device: The device to run the training on.
    Returns:
        avg_loss (float): The average loss over the epoch.
    """
    model.train()
    total_loss = 0
    pbar = tqdm(data_loader, desc="Training", leave=False)
    for batch in pbar:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        _, loss = model(images, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate_epoch(model, data_loader, device) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        model: The model to validate.
        data_loader: DataLoader providing the validation data.
        device: The device to run the validation on.
    Returns:
        avg_loss (float): The average loss over the validation set.
        val_map (float): The balanced mean Average Precision (mAP) over the validation set.
    """
    model.eval()
    total_loss = 0
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Validation", leave=False)
        for batch in pbar:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            embeddings, loss = model(images, labels)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

            total_loss += loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(data_loader)
    avl_embeddings = torch.cat(all_embeddings, dim=0)
    avl_labels = torch.cat(all_labels, dim=0)
    val_map = compute_validation_map(avl_embeddings, avl_labels.cpu().numpy())
    return avg_loss, val_map


def compute_validation_map(embeddings: torch.Tensor, val_labels: np.ndarray) -> float:
    """
    Optimized version of v1 logic.
    Matches the 'balanced' identity averaging of your winning version.

    Args:
        embeddings (torch.Tensor): Embeddings of the validation samples.
        val_labels (np.ndarray): Ground truth labels for the validation samples.
    Returns:
        balanced_map (float): The balanced mean Average Precision (mAP).
    """
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())

    sim_matrix = sim_matrix.cpu().numpy()
    np.fill_diagonal(sim_matrix, -1)

    query_aps = {}

    for query_idx in range(len(val_labels)):
        query_label = val_labels[query_idx]
        similarities = sim_matrix[query_idx]

        is_match = (val_labels == query_label).astype(int)
        is_match[query_idx] = 0

        n_positives = is_match.sum()
        if n_positives == 0:
            continue

        sorted_indices = np.argsort(-similarities)
        sorted_matches = is_match[sorted_indices]

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
