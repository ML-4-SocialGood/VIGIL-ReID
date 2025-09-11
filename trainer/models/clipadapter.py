import os

import torch
import torch.nn as nn
from clip import clip
from torch.nn import functional as F

from metrics import compute_accuracy
from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from utils import PROMPT_TEMPLATES
from loss.make_loss import make_loss


class Adapter(nn.Module):
    def __init__(self, channel_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel_in, channel_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_in // reduction, channel_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        if self.cfg.MODEL.CLIPAdapter.BACKBONE == "RN50":
            self.adapter = Adapter(1024, 4)
        elif self.cfg.MODEL.CLIPAdapter.BACKBONE == "ViT-B/32":
            self.adapter = Adapter(512, 4)
        self.dtype = clip_model.dtype

        prompt_template = PROMPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = [
            # currently using animal ID directly in the prompt
            prompt_template.format(str(class_name).replace("_", " "))
            for class_name in class_names
        ]
        prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts])
        prompts = prompts.to(torch.cuda.current_device())

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            # Normalize with epsilon and keep computations in float32 for stability
            text_features = text_features.float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            self.text_features = text_features

    def forward(self, image):
        adapter_ratio = 0.8
        image_features = self.image_encoder(image.type(self.dtype))
        # Run adapter and mixing in float32 for numerical stability
        base_features = image_features.float()
        adapter_features = self.adapter(base_features)
        mixed_features = (
            adapter_ratio * adapter_features + (1 - adapter_ratio) * base_features
        )

        # Normalize safely in float32
        norm = mixed_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        mixed_features_norm = mixed_features / norm

        # Stabilize logit scaling and similarity computation
        logit_scale = self.logit_scale.float().exp().clamp(max=100.0)
        logits = logit_scale * mixed_features_norm @ self.text_features.float().t()

        # Return logits and unnormalized float32 features for metric losses
        return logits, mixed_features


@MODEL_REGISTRY.register()
class CLIPAdapter(Trainer):
    """CLIP-Adapter

    CLIP-Adapter: Better Vision-Language Models with Feature Adapters
    https://arxiv.org/abs/2110.04544
    """

    def build_model(self):
        print("Loading CLIP Backbone: {}".format(self.cfg.MODEL.CLIPAdapter.BACKBONE))
        clip_model, _ = clip.load(
            self.cfg.MODEL.CLIPAdapter.BACKBONE,
            device=self.device,
            download_root=os.path.abspath(os.path.expanduser("data")),
        )

        print("Building Custom CLIP")
        self.model = CustomCLIP(
            self.cfg, self.data_manager.dataset.class_names, clip_model
        )

        print("Turning Off Gradients in Image and Text Encoder")
        for name, param in self.model.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)

        # Double check
        enabled_params = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled_params.add(name)
        print("Parameters to be updated: {}".format(enabled_params))

        self.model.to(self.device)

        # NOTE: Only Give self.model.adapter to the Optimizer
        self.optimizer = build_optimizer(self.model.adapter, self.cfg.OPTIM)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)

        self.model_registeration(
            "clip_adapter",
            self.model.adapter,
            self.optimizer,
            self.lr_scheduler,
        )

    def forward_backward(self, batch_data):
        image, target, domain = self.parse_batch_train(batch_data)
        output, feat = self.model(image)
        loss, center_criterion = make_loss(self.cfg, self.num_classes, self.device)
        loss = loss(output, feat, target, None)

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            # "acc": compute_accuracy(output, target)[0].item(),
        }

        # LR scheduler now steps once per epoch in Trainer.after_epoch

        return loss_summary
