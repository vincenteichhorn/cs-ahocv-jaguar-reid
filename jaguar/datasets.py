from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict

from jaguar.embedding_models import EmbeddingModel


def get_data(data_path: str, validation_split_size=0.2, seed=42):
    """Get dataset located at `path`.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, LabelEncoder]: Training data, validation data, test data,
        number of classes, and label encoder.
    """
    data_path = Path(data_path) if isinstance(data_path, str) else data_path
    train_df = pd.read_csv(data_path / "train.csv")
    train_df["filename"] = train_df["filename"].apply(lambda x: str(data_path / "train/train" / x))
    num_classes = train_df["ground_truth"].nunique()
    label_encoder = LabelEncoder()
    train_df["label_encoded"] = label_encoder.fit_transform(train_df["ground_truth"])
    train_data, val_data = train_test_split(
        train_df,
        test_size=validation_split_size,
        random_state=seed,
        stratify=train_df["ground_truth"],
    )

    test_data = pd.read_csv(data_path / "test.csv")
    test_a_data = test_data.copy()
    test_a_data["filename"] = test_a_data["query_image"].apply(lambda x: str(data_path / "test/test" / x))
    test_a_data["label_encoded"] = -1  # Dummy labels for test set
    test_b_data = test_data.copy()
    test_b_data["filename"] = test_b_data["gallery_image"].apply(lambda x: str(data_path / "test/test" / x))
    test_b_data["label_encoded"] = -1  # Dummy labels for test set

    return train_data, val_data, test_a_data, test_b_data, num_classes, label_encoder


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        embedding_model: EmbeddingModel,
        key="train",
        image_column="filename",
        label_column="label_encoded",
        cache_folder="./embeddings",
    ):
        self.data = data
        self.image_column = image_column
        self.label_column = label_column

        self.unique_filepaths = sorted(data[image_column].unique().tolist())
        self.unique_filenames = [f.split("/")[-1] for f in self.unique_filepaths]
        self.cache_file = Path(cache_folder) / f"{embedding_model.__class__.__name__.lower()}_{key}_embeddings.npz"
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        unique_embeddings = self._get_unique_embeddings(embedding_model)

        self.fn_to_idx = {fn: i for i, fn in enumerate(self.unique_filepaths)}
        self.unique_embeddings = torch.from_numpy(unique_embeddings)
        self.all_embeddings = self.get_embeddings()

        self.labels = data[label_column].values

    def _get_unique_embeddings(self, embedding_model):
        """Handles the logic of loading from cache or extracting fresh."""
        if self.cache_file.exists():
            print(f"Loading cached embeddings from {self.cache_file}")
            with np.load(self.cache_file, allow_pickle=True) as z_file:
                cached_fns = [f.split("/")[-1] for f in list(z_file["filenames"])]

                if cached_fns == self.unique_filenames:
                    return z_file["embeddings"]
                else:
                    print("Cache mismatch (unique filenames changed). Re-extracting...")

        print(f"Extracting embeddings for {len(self.unique_filepaths)} unique images...")
        embeddings = embedding_model.extract_embeddings(self.unique_filepaths, batch_size=32, verbose=True)

        emb_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings

        np.savez_compressed(
            self.cache_file,
            embeddings=emb_np,
            filenames=np.array(self.unique_filenames, dtype=object),
        )
        return emb_np

    def get_embeddings(self):
        return torch.stack([self.unique_embeddings[self.fn_to_idx[fn]] for fn in self.data[self.image_column]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        embedding = self.all_embeddings[idx]
        label = self.labels[idx]

        return embedding, label


class PKSampler(Sampler):
    def __init__(self, labels, p, k):
        self.p = p
        self.k = k
        self.labels = np.array(labels)
        self.unique_labels = np.unique(self.labels)

        # Group indices by label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        # Filter out classes that don't have enough samples to satisfy K
        self.valid_labels = [lbl for lbl in self.unique_labels if len(self.label_to_indices[lbl]) >= k]

    def __iter__(self):
        # Calculate how many full PK batches we can create
        num_batches = len(self.labels) // (self.p * self.k)

        for _ in range(num_batches):
            # Select P identities
            selected_labels = np.random.choice(self.valid_labels, self.p, replace=False)

            indices = []
            for lbl in selected_labels:
                # Select K samples for each identity
                targ_indices = self.label_to_indices[lbl]
                indices.extend(np.random.choice(targ_indices, self.k, replace=False))

            # Yielding indices in a single batch
            yield indices

    def __len__(self):
        return len(self.labels) // (self.p * self.k)
