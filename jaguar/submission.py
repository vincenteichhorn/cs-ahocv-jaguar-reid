from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F


def build_submission(submission_path, model, test_dataloader, device):

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

            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

    sim_matrix = torch.mm(all_embeddings, all_embeddings.t())
    sim_matrix.fill_diagonal_(-1)
    sim_matrix = sim_matrix.clip(0.0, 1.0).cpu().numpy()

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
