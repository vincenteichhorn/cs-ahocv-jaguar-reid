from logging import config
from pathlib import Path
from typing import List, Union
import timm
from abc import ABC, abstractmethod
from PIL import Image

import torch
from torchvision import transforms
from tqdm import tqdm


class EmbeddingModel(ABC):

    def __init__(self, device: str):

        self.device = device
        self.model = self.load_model().to(self.device)
        self.model.eval()
        self.model.to(self.device)

    @abstractmethod
    def load_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def get_transforms(self) -> transforms.Compose:
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def extract_embeddings(self, images: List[Union[str, Path]], batch_size: int = 32, verbose=True) -> torch.Tensor:

        transforms = self.get_transforms()
        embeddings = []

        for i in tqdm(range(0, len(images), batch_size), disable=not verbose, desc="Extracting embeddings"):
            batch_images = images[i : i + batch_size]
            # Load and preprocess images
            batch_tensors = []
            for img_path in batch_images:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transforms(img)
                batch_tensors.append(img_tensor)

            input_batch = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                embeddings_batch = self.forward(input_batch)
            embeddings.append(embeddings_batch.cpu())

        return torch.cat(embeddings, dim=0)


class MegaDescriptor(EmbeddingModel):

    def __init__(self, device: str, input_size: int = 384, get_feature_map: bool = False):
        self.input_size = input_size
        self.get_feature_map = get_feature_map
        super().__init__(device)

    def load_model(self) -> torch.nn.Module:
        if self.get_feature_map:
            return timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0, global_pool="")
        return timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)

    def get_transforms(self) -> transforms.Compose:
        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        return transforms

    def get_embedding_dim(self) -> int:
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        dummy_output = self.model(dummy_input)
        return dummy_output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.get_feature_map:
            output = self.model(x)  # shape (B, H, W, C)
            output = output.permute(0, 3, 1, 2)  # shape (B, C, H, W)
            return output
        return self.model(x)


class DINOv3(EmbeddingModel):

    def __init__(self, device: str, input_size: int = 256, get_feature_map: bool = False):
        self.input_size = input_size
        self.get_feature_map = get_feature_map
        super().__init__(device)

    def load_model(self) -> torch.nn.Module:
        if self.get_feature_map:
            return timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, global_pool="")
        return timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True)

    def get_transforms(self) -> transforms.Compose:
        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        return transforms

    def get_embedding_dim(self) -> int:
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        dummy_output = self.model(dummy_input)
        return dummy_output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.get_feature_map:
            output = self.model(x)
            B, N, C = output.shape
            patches = output[:, 5:, :]
            num_patches = patches.shape[1]
            grid_size = int(num_patches**0.5)
            feature_map = patches.transpose(1, 2).reshape(B, C, grid_size, grid_size)
            return feature_map  # shape (B, C, H, W)
        return self.model(x)


class EfficientNetB4(EmbeddingModel):

    def __init__(self, device: str, input_size: int = 380, get_feature_map: bool = False):
        self.input_size = input_size
        self.get_feature_map = get_feature_map
        super().__init__(device)

    def load_model(self) -> torch.nn.Module:
        if self.get_feature_map:
            return timm.create_model("efficientnet_b4", pretrained=True, num_classes=0, global_pool="")
        return timm.create_model("efficientnet_b4", pretrained=True, num_classes=0)

    def get_transforms(self) -> transforms.Compose:
        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        return transforms

    def get_embedding_dim(self) -> int:
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        dummy_output = self.model(dummy_input)
        return dummy_output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MobileNetV3(EmbeddingModel):

    def __init__(self, device: str, input_size: int = 224, get_feature_map: bool = False):
        self.input_size = input_size
        self.get_feature_map = get_feature_map
        super().__init__(device)

    def load_model(self) -> torch.nn.Module:
        if self.get_feature_map:
            return timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=0, global_pool="")
        return timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=0)

    def get_transforms(self) -> transforms.Compose:
        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        return transforms

    def get_embedding_dim(self) -> int:
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        dummy_output = self.model(dummy_input)
        return dummy_output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
