# import os

# import torch
# import torch.nn as nn
# from .dino_utils import create_linear_input
# from torch.nn import functional as F
# from transformers import AutoModel, AutoProcessor, SiglipVisionModel
# from metrics import compute_accuracy
# from optim import build_lr_scheduler, build_optimizer
# from trainer import MODEL_REGISTRY, Trainer
# from utils import PROMPT_TEMPLATES
# from loss.make_loss import make_loss


# class Adapter(nn.Module):
#     def __init__(self, channel_in, reduction=4):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(channel_in, channel_in // reduction, bias=False),
#             nn.ReLU(),  
#             nn.Dropout(0.5),
#             nn.Linear(channel_in // reduction, channel_in, bias=False),
#             nn.ReLU(),  
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         return x


# class CustomDino(nn.Module):
#     def __init__(self, cfg, num_classes, dino_model, domains):
#         super().__init__()
#         self.cfg = cfg
#         self.dino_model = dino_model

#         output_dim = self.dino_model.embed_dim
#         self.adapter_dict = nn.ModuleDict()
#         self.classifier_dict = nn.ModuleDict()
#         for i, domain in enumerate(domains):
#             self.adapter_dict[f"adapter_{i}"] = Adapter(output_dim, 4)
#             self.classifier_dict[f"classifier_{i}"] = nn.Linear(output_dim, num_classes[i])
#         self.num_classes = num_classes
#         self.domains = domains

#     def forward(self, image, domains):
#         adapter_ratio = 0.4  
#         # Get cls token from DINO
#         x_tokens_list = self.dino_model.get_intermediate_layers(image, n=1, return_class_token=True)
#         image_features = create_linear_input(x_tokens_list, 1, False)
#         image_features = torch.nn.functional.normalize(image_features, dim=-1, eps=1e-6)
#         base_features = image_features.float()

#         # Check if all samples in batch have the same domain (training case)
#         unique_domains = list(set(domains))
#         single_domain_batch = len(unique_domains) == 1
        
#         mixed_features = torch.zeros_like(base_features)
        
#         if single_domain_batch:
#             # Training case: all samples have same domain, can use unified tensor
#             domain = domains[0]
#             cls_scores = torch.zeros(base_features.size(0), self.num_classes[domain], 
#                                     device=base_features.device, dtype=base_features.dtype)
            
#             # iterate through all images in the batch
#             for i, domain_id in enumerate(domains):
#                 # Get the feature for this specific image
#                 feature = base_features[i:i+1]  # Keep batch dimension: [1, proj_dim]
                
#                 # Apply the corresponding adapter
#                 adapter_features = self.adapter_dict[f"adapter_{domain_id}"](feature)
#                 mixed_feature = (
#                     adapter_ratio * adapter_features + (1 - adapter_ratio) * feature
#                 )
                
#                 # Put mixed feature back
#                 mixed_features[i] = mixed_feature.squeeze(0)
                
#                 # Use domain-specific classifier for each sample
#                 cls_score = self.classifier_dict[f"classifier_{domain_id}"](mixed_feature)
#                 cls_scores[i] = cls_score.squeeze(0)
#         else:
#             # Inference: doesn't use classifier
#             cls_scores = None
#             for i, domain_id in enumerate(domains):
#                 # Get the feature for this specific image
#                 feature = base_features[i:i+1]  # Keep batch dimension: [1, proj_dim]
                
#                 # Apply the corresponding adapter
#                 adapter_features = self.adapter_dict[f"adapter_{domain_id}"](feature)
#                 mixed_feature = (
#                     adapter_ratio * adapter_features + (1 - adapter_ratio) * feature
#                 )
                
#                 # Put mixed feature back
#                 mixed_features[i] = mixed_feature.squeeze(0)
                

#         # Normalize mixed features for retrieval/metric learning
#         mixed_features_norm = torch.nn.functional.normalize(mixed_features, dim=-1, eps=1e-6)

#         # Return classifier scores and normalized projected features for metric losses/eval
#         return cls_scores, mixed_features_norm


# @MODEL_REGISTRY.register()
# class DinoAdapterCopy(Trainer):
#     """
#     Dino-Adapter for multi-source domain adaptation
#     """

#     def build_model(self):
#         print("Loading Dino backbone: {}".format(self.cfg.MODEL.DinoAdapter.BACKBONE))
#         dino_model = torch.hub.load(self.cfg.MODEL.DinoAdapter.REPO, self.cfg.MODEL.DinoAdapter.BACKBONE, source='local', 
#                                     weights=self.cfg.MODEL.DinoAdapter.WEIGHT_PATH)
#         self.dino_model = dino_model.to(self.device)

#         print("Building Custom Dino")
#         self.model = CustomDino(
#             self.cfg, self.data_manager.num_classes, dino_model, self.data_manager.get_target_domains
#         )

#         print("Turning Off Gradients in Image and Text Encoder")
#         for name, param in self.model.named_parameters():
#             if "adapter_dict" not in name and "classifier_dict" not in name:
#                 param.requires_grad_(False)

#         # Double check
#         enabled_params = set()
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 enabled_params.add(name)
#         print("Parameters to be updated: {}".format(enabled_params))

#         self.model.to(self.device)

#         # Create domain-specific optimizers and schedulers
#         self.domain_optimizers = {}
#         self.domain_schedulers = {}
        
#         # use target domains which is a subset of source domains
#         for domain_id, domain_name in enumerate(self.data_manager.get_target_domains):
#             # Get domain-specific parameters (adapter + classifier for this domain)
#             domain_params = []
#             domain_params.extend(list(self.model.adapter_dict[f"adapter_{domain_id}"].parameters()))
#             domain_params.extend(list(self.model.classifier_dict[f"classifier_{domain_id}"].parameters()))
            
#             # Get domain-specific learning rate multiplier if available
#             lr_multiplier = 1.0
#             if hasattr(self.cfg.OPTIM, 'DOMAIN_OPTIM') and hasattr(self.cfg.OPTIM.DOMAIN_OPTIM, 'DOMAIN_LR_MULTIPLIERS'):
#                 multipliers = self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_LR_MULTIPLIERS
#                 if isinstance(multipliers, (list, tuple)) and len(multipliers) > domain_id:
#                     lr_multiplier = multipliers[domain_id]
            
#             # Create domain-specific optimizer with custom learning rate
#             domain_optimizer = build_optimizer(domain_params, self.cfg.OPTIM, lr_multiplier=lr_multiplier)
#             self.domain_optimizers[domain_id] = domain_optimizer
            
#             # Create domain-specific scheduler with custom config if available
#             domain_scheduler = self._build_domain_scheduler(domain_optimizer, domain_id)
#             self.domain_schedulers[domain_id] = domain_scheduler
            
#             # Register each domain optimizer and scheduler
#             self.model_registeration(
#                 f"siglip_adapter_domain_{domain_id}",
#                 self.model,
#                 domain_optimizer,
#                 domain_scheduler,
#             )
            
#             # Print domain-specific configuration info
#             scheduler_type = "default"
#             if (hasattr(self.cfg.OPTIM, 'DOMAIN_OPTIM') and
#                 hasattr(self.cfg.OPTIM.DOMAIN_OPTIM, 'DOMAIN_SCHEDULERS') and
#                 len(self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_SCHEDULERS) > domain_id):
#                 scheduler_type = self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_SCHEDULERS[domain_id]
#             print(f"Domain {domain_id}: LR base={self.cfg.OPTIM.LR:.6f}, mult={lr_multiplier:.3f}, Scheduler={scheduler_type}")
            

#     def forward_backward(self, batch_data):
#         image, target, domains, time = self.parse_batch_train(batch_data)
#         # all samples in the batch have the same domain
#         domain = domains[0]
#         output, feat = self.model(image, domains)
#         loss_func, center_criterion = make_loss(self.cfg, self.num_classes[domain], self.device)
#         loss, ID_LOSS, TRI_LOSS = loss_func(output, feat, target, None)

#         # Use domain-specific optimizer for backward and update
#         self.model_backward_and_update(loss, f"siglip_adapter_domain_{domain}")

#         loss_summary = {
#             "domain": self.data_manager.get_source_domains[domain],
#             "domain_id": domain,
#             "loss": loss.item(),
#             "ID_LOSS": ID_LOSS,
#             "TRI_LOSS": TRI_LOSS,
#             "acc": compute_accuracy(output, target)[0].item(),
#         }

#         # LR scheduler now steps once per epoch in Trainer.after_epoch

#         return loss_summary

#     def model_inference(self, batch_data, domains):
#         _, feat = self.model(batch_data, domains)
#         return feat
    
#     def _build_domain_scheduler(self, optimizer, domain_id):
#         """Build domain-specific learning rate scheduler with custom config if available."""
#         # Check if domain-specific scheduler config exists
#         if hasattr(self.cfg.OPTIM, 'DOMAIN_OPTIM') and hasattr(self.cfg.OPTIM.DOMAIN_OPTIM, 'DOMAIN_SCHEDULERS'):
#             domain_schedulers = self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_SCHEDULERS
#             if isinstance(domain_schedulers, (list, tuple)) and len(domain_schedulers) > domain_id:
#                 scheduler_type = domain_schedulers[domain_id]
                
#                 # Get domain-specific step size if using StepLR
#                 step_size = None
#                 if (scheduler_type == "StepLR" and 
#                     hasattr(self.cfg.OPTIM.DOMAIN_OPTIM, 'DOMAIN_STEP_SIZES') and
#                     isinstance(self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_STEP_SIZES, (list, tuple)) and
#                     len(self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_STEP_SIZES) > domain_id):
#                     step_size = self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_STEP_SIZES[domain_id]
                
#                 return build_lr_scheduler(optimizer, self.cfg.OPTIM, 
#                                         scheduler_type=scheduler_type, step_size=step_size)
        
#         # Fall back to default scheduler
#         return build_lr_scheduler(optimizer, self.cfg.OPTIM)