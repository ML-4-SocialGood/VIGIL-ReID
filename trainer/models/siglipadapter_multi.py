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
            nn.Dropout(0.5),
            nn.Linear(channel_in // reduction, channel_in, bias=False),
            nn.GELU(),  # Changed from ReLU to GELU
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomSIGLIP(nn.Module):
    def __init__(self, cfg, num_classes, siglip_model, domains):
        super().__init__()
        self.cfg = cfg
        self.siglip_model = siglip_model

        proj_dim = siglip_model.config.text_config.projection_size
        self.adapter_dict = nn.ModuleDict()
        self.classifier_dict = nn.ModuleDict()
        for i, domain in enumerate(domains):
            self.adapter_dict[f"adapter_{i}"] = Adapter(proj_dim, 4)
            self.classifier_dict[f"classifier_{i}"] = nn.Linear(proj_dim, num_classes)
        self.dtype = siglip_model.dtype
        self.num_classes = num_classes


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

    def forward(self, image, domains):
        adapter_ratio = 0.4  
        # Get projected image features directly from SigLIP (already projected!)
        image_features = self.siglip_model.get_image_features(**{'pixel_values': image})
        base_features = image_features.float()
        
        # Process each sample individually using its corresponding domain
        mixed_features = torch.zeros_like(base_features)
        cls_scores = torch.zeros(base_features.size(0), self.num_classes, 
                                device=base_features.device, dtype=base_features.dtype)
        
        for i, domain_id in enumerate(domains):
            # Get the feature for this specific image
            feature = base_features[i:i+1]  # Keep batch dimension: [1, proj_dim]
            
            # Apply the corresponding adapter
            adapter_features = self.adapter_dict[f"adapter_{domain_id}"](feature)
            mixed_feature = (
                adapter_ratio * adapter_features + (1 - adapter_ratio) * feature
            )
            
            # Put mixed feature back
            mixed_features[i] = mixed_feature.squeeze(0)
            
            # Use domain-specific classifier for each sample (no additional projection needed)
            cls_score = self.classifier_dict[f"classifier_{domain_id}"](mixed_feature)
            cls_scores[i] = cls_score.squeeze(0)

        # Normalize mixed features for retrieval/metric learning
        mixed_features_norm = torch.nn.functional.normalize(mixed_features, dim=-1, eps=1e-6)

        # Return classifier scores and normalized projected features for metric losses/eval
        return cls_scores, mixed_features_norm


@MODEL_REGISTRY.register()
class SIGLIPAdapter_multi(Trainer):
    """
    SIGLIP-Adapter for multi-source domain adaptation
    """

    def build_model(self):
        print("Loading SIGLIP Checkpoint: {}".format(self.cfg.MODEL.SIGLIPAdapter.CKPT))
        siglip_model= AutoModel.from_pretrained(
            self.cfg.MODEL.SIGLIPAdapter.CKPT,
        )
        self.siglip_model = siglip_model.to(self.device)

        print("Building Custom SIGLIP")
        self.model = CustomSIGLIP(
            self.cfg, self.data_manager.dataset.num_classes, siglip_model, self.data_manager.get_source_domains
        )

        print("Turning Off Gradients in Image and Text Encoder")
        for name, param in self.model.named_parameters():
            if "adapter_dict" not in name and "classifier_dict" not in name:
                param.requires_grad_(False)

        # Double check
        enabled_params = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled_params.add(name)
        print("Parameters to be updated: {}".format(enabled_params))

        self.model.to(self.device)

        # NOTE: Give both adapter_dict and classifier_dict to the Optimizer
        trainable_params = list(self.model.adapter_dict.parameters()) + list(self.model.classifier_dict.parameters())
        self.optimizer = build_optimizer(trainable_params, self.cfg.OPTIM)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)

        self.model_registeration(
            "siglip_adapter",
            self.model,  # Register the full model
            self.optimizer,
            self.lr_scheduler,
        )

    def forward_backward(self, batch_data):
        image, target, domains = self.parse_batch_train(batch_data)
        output, feat = self.model(image, domains)
        loss_func, center_criterion = make_loss(self.cfg, self.num_classes, self.device)
        loss, ID_LOSS, TRI_LOSS = loss_func(output, feat, target, None)

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "ID_LOSS": ID_LOSS,
            "TRI_LOSS": TRI_LOSS,
            "acc": compute_accuracy(output, target)[0].item(),
        }

        # LR scheduler now steps once per epoch in Trainer.after_epoch

        return loss_summary

    def model_inference(self, batch_data):
        print("Inferring on the model")