import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor, SiglipVisionModel
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
            nn.GELU(),  # Changed from ReLU to GELU for better gradient flow
            nn.Linear(channel_in // reduction, channel_in, bias=False),
            nn.GELU(),  # Changed from ReLU to GELU
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomSIGLIP(nn.Module):
    def __init__(self, cfg, num_classes, siglip_model):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = siglip_model.vision_model
        # self.logit_scale = siglip_model.logit_scale

        feature_dim = self.image_encoder.config.hidden_size
        self.adapter = Adapter(feature_dim, 4)
        self.dtype = siglip_model.dtype

        # Use SigLIP's visual projection so our features align with zero-shot
        self.proj_layer = getattr(siglip_model, "visual_projection", None)
        if self.proj_layer is None:
            # Fallback to identity if projection is absent (should not happen for SigLIP)
            self.proj_layer = nn.Identity()
            proj_dim = feature_dim
        else:
            proj_dim = getattr(self.proj_layer, "out_features", feature_dim)


        # Classifier operates on projected features
        self.classifier = nn.Linear(proj_dim, num_classes)

        # not using text features for now
        # prompt_template = PROMPT_TEMPLATES[cfg.DATASET.NAME]
        # prompts = [
        #     # currently using animal ID directly in the prompt
        #     prompt_template.format(str(class_name).replace("_", " "))
        #     for class_name in class_names
        # ]
        # prompts = torch.cat([siglip_model.tokenize(prompt) for prompt in prompts])
        # prompts = prompts.to(torch.cuda.current_device())

        # with torch.no_grad():
        #     text_features = siglip_model.encode_text(prompts)
        #     # Normalize with epsilon and keep computations in float32 for stability
        #     text_features = text_features.float()
        #     text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        #     self.text_features = text_features

    def forward(self, image):
        adapter_ratio = 0.4  
        image_features = self.image_encoder(image)
        # Extract the pooled output from the model output
        image_features = image_features.pooler_output
        # Run adapter and mixing in float32 for numerical stability
        base_features = image_features.float()
        adapter_features = self.adapter(base_features)
        mixed_features = (
            adapter_ratio * adapter_features + (1 - adapter_ratio) * base_features
        )

        # Project to SigLIP's retrieval space and normalize
        projected = self.proj_layer(mixed_features)
        projected_norm = torch.nn.functional.normalize(projected, dim=-1, eps=1e-6)

        # Classification uses unnormalized projected features (decoupled from retrieval)
        cls_scores = self.classifier(projected)

        # Return classifier scores and normalized projected features for metric losses/eval
        return cls_scores, projected_norm


@MODEL_REGISTRY.register()
class SIGLIPAdapter(Trainer):
    """SIGLIP-Adapter
    """

    def build_model(self):
        print("Loading SIGLIP Checkpoint: {}".format(self.cfg.MODEL.SIGLIPAdapter.CKPT))
        siglip_model= AutoModel.from_pretrained(
            self.cfg.MODEL.SIGLIPAdapter.CKPT,
        )
        self.siglip_model = siglip_model.to(self.device)

        print("Building Custom SIGLIP")
        self.model = CustomSIGLIP(
            self.cfg, self.data_manager.dataset.num_classes, siglip_model
        )

        print("Turning Off Gradients in Image and Text Encoder")
        for name, param in self.model.named_parameters():
            if "adapter" not in name and "classifier" not in name:
                param.requires_grad_(False)

        # Double check
        enabled_params = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled_params.add(name)
        print("Parameters to be updated: {}".format(enabled_params))

        self.model.to(self.device)

        # NOTE: Give both adapter and classifier to the Optimizer
        trainable_params = list(self.model.adapter.parameters()) + list(self.model.classifier.parameters())
        self.optimizer = build_optimizer(trainable_params, self.cfg.OPTIM)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)

        self.model_registeration(
            "siglip_adapter",
            self.model,  # Register the full model
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
            "acc": compute_accuracy(output, target)[0].item(),
        }

        # LR scheduler now steps once per epoch in Trainer.after_epoch

        return loss_summary
