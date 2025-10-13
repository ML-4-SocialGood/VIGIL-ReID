import os
from typing import Iterable, List

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.adamw import AdamW

try:
    from torch.serialization import add_safe_globals
except ImportError:  # Older PyTorch versions do not expose safe globals API
    add_safe_globals = None

from transformers import AutoModel

from trainer import MODEL_REGISTRY, Trainer

from .siglipadapter import Adapter


class CustomSIGLIPAdapterInference(nn.Module):
    def __init__(self, siglip_model: nn.Module, domains: List[str], adapter_ratio: float = 0.4):
        super().__init__()
        self.siglip_model = siglip_model
        self.adapter_ratio = adapter_ratio

        proj_dim = siglip_model.config.text_config.projection_size
        self.adapter_dict = nn.ModuleDict()
        for domain_idx, _ in enumerate(domains):
            self.adapter_dict[f"adapter_{domain_idx}"] = Adapter(proj_dim, 4)

    def forward(self, image: torch.Tensor, domains: Iterable[int]) -> torch.Tensor:
        # Get projected image features directly from SigLIP (already projected)
        image_features = self.siglip_model.get_image_features(**{"pixel_values": image})
        base_features = image_features.float()
        mixed_features = torch.zeros_like(base_features)

        if isinstance(domains, torch.Tensor):
            domain_ids = domains.tolist()
        else:
            domain_ids = list(domains)

        for sample_idx, domain_id in enumerate(domain_ids):
            domain_key = f"adapter_{int(domain_id)}"
            if domain_key not in self.adapter_dict:
                raise KeyError(f"Adapter for domain id {domain_id} not found.")

            feature = base_features[sample_idx : sample_idx + 1]
            adapter_features = self.adapter_dict[domain_key](feature)
            mixed_feature = (
                self.adapter_ratio * adapter_features + (1 - self.adapter_ratio) * feature
            )
            mixed_features[sample_idx] = mixed_feature.squeeze(0)

        return torch.nn.functional.normalize(mixed_features, dim=-1, eps=1e-6)


@MODEL_REGISTRY.register()
class SIGLIPAdapterInference(Trainer):
    """Inference-only SigLIP adapter that loads trained adapters without classifiers."""

    def build_model(self):
        adapter_cfg = self.cfg.MODEL.SIGLIPAdapterInference
        siglip_ckpt = adapter_cfg.CKPT
        adapter_ckpt = "/home/clou785/VIGIL-ReID/output/model_cat.pth.tar-25"
        adapter_ratio = adapter_cfg.ADAPTER_RATIO

        if not adapter_ckpt:
            raise ValueError("cfg.MODEL.SIGLIPAdapterInference.ADAPTER_WEIGHTS must be provided for inference.")
        if not os.path.isfile(adapter_ckpt):
            raise FileNotFoundError(f"Adapter checkpoint not found at {adapter_ckpt}")

        print(f"Loading SIGLIP checkpoint: {siglip_ckpt}")
        siglip_model = AutoModel.from_pretrained(siglip_ckpt)
        siglip_model = siglip_model.to(self.device)

        print("Building inference-only SIGLIP adapter")
        self.model = CustomSIGLIPAdapterInference(
            siglip_model,
            self.data_manager.get_target_domains,
            adapter_ratio=adapter_ratio,
        )
        self.model.to(self.device)

        self._load_adapter_weights(adapter_ckpt)

        for param in self.model.parameters():
            param.requires_grad_(False)

        self.model_registeration("siglip_adapter_inference", self.model, None, None)

    def forward_backward(self, batch_data):
        raise RuntimeError("SIGLIPAdapterInference is inference-only and does not support training.")

    def model_inference(self, batch_data, domains):
        return self.model(batch_data, domains)

    def _load_adapter_weights(self, adapter_ckpt: str) -> None:
        if add_safe_globals is not None:
            # Allow CosineAnnealingLR to be deserialized when weights_only=True (PyTorch >= 2.6)
            add_safe_globals([CosineAnnealingLR])

        checkpoint = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        adapter_state = {
            key.replace("adapter_dict.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("adapter_dict.")
        }

        load_result = self.model.adapter_dict.load_state_dict(adapter_state, strict=False)

        if load_result.missing_keys:
            print(f"Warning: missing adapter keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Warning: unexpected adapter keys: {load_result.unexpected_keys}")
