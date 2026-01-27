import torch
from tqdm import tqdm


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for embeddings, labels in pbar:
        embeddings, labels = embeddings.to(device), labels.to(device)

        # Forward pass
        model_embeddings = model(embeddings)
        loss = criterion(model_embeddings, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    return avg_loss


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for embeddings, labels in pbar:
            embeddings, labels = embeddings.to(device), labels.to(device)

            model_embeddings = model(embeddings)
            loss = criterion(model_embeddings, labels)

            total_loss += loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    return avg_loss
