import gc
from pathlib import Path
from typing import Callable, Dict, Literal, Tuple
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def get_data(data_path: str, validation_split_size=0.2, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, LabelEncoder]:
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

    test_data_all = pd.read_csv(data_path / "test.csv")
    test_data = test_data_all.copy().drop(columns=["gallery_image"])
    test_data = test_data.drop_duplicates(subset=["query_image"]).reset_index(drop=True)
    test_data["filename"] = test_data["query_image"].apply(lambda x: str(data_path / "test/test" / x))
    test_data["label_encoded"] = -1  # Dummy labels for test set

    return train_data, val_data, test_data, num_classes, label_encoder


def get_dataloaders(
    data_dir: str,
    validation_split_size: float,
    seed: int,
    batch_size: int,
    image_size: int,
    cache_dir: str,
    train_process_fn: Callable = None,
    val_process_fn: Callable = None,
    mode: Literal["segmented", "background"] = "background",
) -> Tuple[DataLoader, DataLoader, DataLoader, int, LabelEncoder]:
    """
    Get dataloaders for training, validation, test query, and test gallery datasets.

    Args:
        data_dir (str): Path to the dataset.
        validation_split_size (float): Proportion of the dataset to include in the validation split.
        seed (int): Random seed for reproducibility.
        batch_size (int): Batch size for the dataloaders.
        image_size (int): Size to which images will be resized.
        cache_dir (str): Directory to cache processed images.
        process_fn (Callable, optional): Function to process images. Defaults to None.
        mode (Literal["segmented", "background"], optional): Mode for image processing. Defaults to "segmented".
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int, LabelEncoder]: Training dataloader, validation dataloader,
        test dataloader, number of classes, and label encoder.
    """
    train_data, val_data, test_gallery, num_classes, label_encoder = get_data(
        data_dir,
        validation_split_size=validation_split_size,
        seed=seed,
    )

    train_dataset = ImageDataset(train_data, prewarm=True, image_size=image_size, cache_dir=cache_dir, key="train", process_fn=train_process_fn, mode=mode)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=12, prefetch_factor=2, persistent_workers=True
    )

    validation_dataset = ImageDataset(val_data, prewarm=True, image_size=image_size, cache_dir=cache_dir, key="val", process_fn=val_process_fn, mode=mode)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True
    )

    test_dataset = ImageDataset(test_gallery, prewarm=True, image_size=image_size, cache_dir=cache_dir, key="test", process_fn=val_process_fn, mode=mode)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True)
    return train_dataloader, validation_dataloader, test_dataloader, num_classes, label_encoder


def get_base_transform():
    base_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return base_transform


def get_resize_transform(image_size: int):
    resize_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
        ]
    )
    return resize_transform


class ImageDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        image_column: str = "filename",
        label_column: str = "label_encoded",
        image_size: int = 384,
        process_fn: Callable = None,
        mode: Literal["segmented", "background"] = "segmented",
        cache_dir: str = "./cache",
        key: str = "train",
        prewarm: bool = True,
        verbose: bool = True,
    ):
        self.data = data
        self.image_column = image_column
        self.image_paths = data[image_column].tolist()
        self.label_column = label_column
        self.labels = data[label_column].tolist()
        self.process_fn = process_fn if process_fn is not None else get_base_transform()
        self.resize_transform = get_resize_transform(image_size)
        self.mode = mode
        self.verbose = verbose

        # Initialize Cache
        self.cache_dir = Path(cache_dir) / f"{key}_{mode}_{image_size}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.to_tensor = transforms.ToTensor()

        if prewarm:
            self._prewarm()

    def _prewarm(self):
        if not all(self._get_cache_path(p).exists() for p in self.image_paths):
            prewarm_loader = DataLoader(
                self,
                batch_size=64,
                shuffle=False,
                num_workers=4,
            )
            for _ in tqdm(prewarm_loader, desc=f"Prewarming cache for {self.cache_dir.name} dataset", total=len(prewarm_loader), disable=not self.verbose):
                pass

    def _get_cache_path(self, image_path: str) -> Path:
        source_path = Path(image_path)
        cache_file = self.cache_dir / f"{source_path.stem}.pt"
        return cache_file

    def load_image(self, image_path: str) -> torch.Tensor:
        cache_file = self._get_cache_path(image_path)

        if cache_file.exists():
            try:
                return torch.load(cache_file, weights_only=True)
            except (UnidentifiedImageError, OSError):
                # If the cache is corrupt, remove it and fall through to recreation
                cache_file.unlink(missing_ok=True)

        # Re-create the image from source

        if self.mode == "segmented":
            img = Image.open(image_path).convert("RGBA")
            black_bg = Image.new("RGB", img.size, (0, 0, 0))
            black_bg.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            processed_img = self.resize_transform(black_bg)
        else:
            img = Image.open(image_path).convert("RGB")
            img = self.resize_transform(img)
            processed_img = img

        tensor_img = self.to_tensor(processed_img)
        torch.save(tensor_img, cache_file, _use_new_zipfile_serialization=False)
        return tensor_img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = self.load_image(image_path)

        if self.process_fn is not None:
            image = self.process_fn(image)

        return image, label
