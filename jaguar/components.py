from pathlib import Path
from typing import Tuple, Union
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


class VielleichtguarModel(nn.Module):
    def __init__(self, backbone: nn.Module, layers: nn.Sequential, criterion: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.criterion = criterion

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        backbone_features = self.backbone(x)
        embeddings = self.layers(backbone_features)
        if labels is not None:
            loss = self.criterion(embeddings, labels)
            return embeddings, loss
        return embeddings

    def save_model(self, path: str, with_backbone: bool = True, with_criterion: bool = True):
        state = {
            "layers": self.layers.state_dict(),
        }
        if with_backbone:
            state["backbone"] = self.backbone.state_dict()
        if with_criterion:
            state["criterion"] = self.criterion.state_dict()
        torch.save(state, path)

    def load_model(self, path: str):
        state = torch.load(path, map_location="cpu")
        if "backbone" in state:
            self.backbone.load_state_dict(state["backbone"])
        self.layers.load_state_dict(state["layers"])
        if "criterion" in state:
            self.criterion.load_state_dict(state["criterion"])


class EmbeddingProjection(nn.Module):

    def __init__(self, input_dim=1536, hidden_dim=512, output_dim=256, dropout=0.3, n_layers=2):
        super().__init__()

        modules = []
        if n_layers >= 2:
            for i in range(n_layers - 1):
                modules.extend(
                    [
                        nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                    ]
                )
        modules.extend(
            [
                nn.Linear(hidden_dim if n_layers >= 2 else input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            ]
        )

        self.embedding_projection = nn.Sequential(*modules)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.embedding_projection(x)


class HypersphericalProjection(nn.Module):
    """
    Projects embeddings onto the unit hypersphere via L2 normalization.
    This makes the embedding space explicitly hyperspherical, where cosine
    similarity is the natural distance metric.
    """

    def __init__(self, input_dim=1536, hidden_dim=512, output_dim=256, dropout=0.3, n_layers=2):
        super().__init__()
        self.projection = EmbeddingProjection(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            n_layers=n_layers,
        )

    def forward(self, x):
        embeddings = self.projection(x)
        return F.normalize(embeddings, p=2, dim=1)


class HyperbolicProjection(nn.Module):
    """
    Projects embeddings into the Poincaré ball model of hyperbolic space
    via the exponential map at the origin. Hyperbolic spaces naturally
    capture hierarchical structure, which may benefit re-identification
    tasks with fine-grained identity distinctions.

    The curvature parameter controls the "radius" of the Poincaré ball:
    higher curvature → smaller ball → more hyperbolic distortion.
    """

    def __init__(self, input_dim=1536, hidden_dim=512, output_dim=256, dropout=0.3, n_layers=2, curvature=1.0, max_norm=0.95):
        super().__init__()
        self.projection = EmbeddingProjection(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            n_layers=n_layers,
        )
        self.curvature = curvature
        self.max_norm = max_norm  # clip to stay safely inside the ball

    def exp_map_zero(self, v):
        """Exponential map at the origin of the Poincaré ball."""
        sqrt_c = self.curvature ** 0.5
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)

    def forward(self, x):
        v = self.projection(x)  # tangent vector at origin
        h = self.exp_map_zero(v)
        # Clip norm to stay strictly inside the ball for numerical stability
        h_norm = h.norm(dim=-1, keepdim=True)
        h = torch.where(h_norm > self.max_norm, h * self.max_norm / h_norm, h)
        return h


class EmbeddingModel(nn.Module):

    def __init__(self, model_name: str, freeze=True, cache_folder="./embeddings", use_caching: bool = True, *args, **kwargs):
        super().__init__()
        self.model = timm.create_model(model_name, *args, **kwargs)
        self.device = self._update_device()
        self.cache_folder = Path(cache_folder) / f"{model_name.replace('/', '_')}_{self.__class__.__name__.lower()}"
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.freeze = freeze
        self.use_caching = use_caching
        if freeze:
            self.freeze_weights()

    def get_transforms(self, is_training=False) -> transforms.Compose:
        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=is_training)
        return transforms

    def out_dim(self) -> int:
        self._update_device()
        img = Image.new("RGB", (512, 512), (0, 0, 0))
        transform = self.get_transforms()
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        dummy_output = self.forward(img_tensor)
        return dummy_output.shape[1]

    def to(self, device):
        rt = super().to(device)
        self.device = self._update_device()
        return rt

    def _update_device(self):
        return next(self.model.parameters()).device

    def _tensor_hash(self, x: torch.Tensor) -> int:
        return torch.hash_tensor(x).item()

    def _get_cache_path(self, input_hash: int) -> Path:
        cache_file = self.cache_folder / f"{input_hash}.pt"
        return cache_file

    def freeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.freeze:
            return self.model(x)
        if not self.use_caching:
            with torch.no_grad():
                return self.model(x)

        batch_size = x.shape[0]
        outputs = [None] * batch_size
        indices_to_compute = []

        for i in range(batch_size):
            sample = x[i]
            h = torch.hash_tensor(sample).item()
            cache_path = self.cache_folder / f"{h}.pt"

            if cache_path.exists():
                outputs[i] = torch.load(cache_path, map_location=x.device, weights_only=True)
            else:
                indices_to_compute.append(i)

        if indices_to_compute:
            to_compute_tensor = x[indices_to_compute]

            with torch.no_grad():
                computed_outputs = self.model(to_compute_tensor)

            for j, original_idx in enumerate(indices_to_compute):
                single_output = computed_outputs[j].unsqueeze(0)
                outputs[original_idx] = single_output

                h = torch.hash_tensor(x[original_idx]).item()
                torch.save(single_output, self.cache_folder / f"{h}.pt")

        return torch.cat(outputs, dim=0)


class MegaDescriptor(EmbeddingModel):

    def __init__(self, *args, **kwargs):
        super().__init__(model_name="hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)  # shape (B, C)
        return output


class MegaDescriptorFM(EmbeddingModel):

    def __init__(self, *args, **kwargs):
        super().__init__(model_name="hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0, global_pool="", *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)  # shape (B, H, W, C)
        output = output.permute(0, 3, 1, 2)  # shape (B, C, H, W)
        return output


class DINOv3(EmbeddingModel):

    def __init__(self, *args, **kwargs):
        super().__init__("vit_large_patch16_dinov3.lvd1689m", pretrained=True, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)  # shape (B, C)
        return output


class EfficientNetB4(EmbeddingModel):

    def __init__(self, *args, **kwargs):
        super().__init__("efficientnet_b4", pretrained=True, *args, **kwargs)


class MobileNetV3(EmbeddingModel):

    def __init__(self, *args, **kwargs):
        super().__init__("mobilenetv3_large_100", pretrained=True, *args, **kwargs)


class EVA02(EmbeddingModel):

    def __init__(self, *args, **kwargs):
        super().__init__("timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", pretrained=True, *args, **kwargs)


class GeMPooling(nn.Module):

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        out = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).squeeze(-1).squeeze(-1)
        return out


class DINOv3FM(EmbeddingModel):

    def __init__(self, *args, **kwargs):
        super().__init__("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, global_pool="", *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)  # shape (B, N, C)
        B, _, C = output.shape
        patches = output[:, 5:, :]
        num_patches = patches.shape[1]
        grid_size = int(num_patches**0.5)
        feature_map = patches.transpose(1, 2).reshape(B, C, grid_size, grid_size)
        return feature_map  # shape (B, C, H, W)


class EfficientNetB4FM(EmbeddingModel):

    def __init__(self, *args, **kwargs):
        super().__init__("efficientnet_b4", pretrained=True, num_classes=0, global_pool="", *args, **kwargs)


class MobileNetV3FM(EmbeddingModel):

    def __init__(self, *args, **kwargs):
        super().__init__("mobilenetv3_large_100", pretrained=True, num_classes=0, global_pool="", *args, **kwargs)
