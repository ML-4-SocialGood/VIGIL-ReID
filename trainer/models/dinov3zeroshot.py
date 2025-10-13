import argparse
import logging
import os
import torch
import torch.nn as nn
from .dino_utils import create_linear_input

from trainer import MODEL_REGISTRY, Trainer

from utils import PROMPT_TEMPLATES


@MODEL_REGISTRY.register()
class DinoV3ZeroShot(Trainer):
    def build_model(self):
        # Load model with memory optimization
        self.model = torch.hub.load(self.cfg.MODEL.DinoV3ZeroShot.REPO, self.cfg.MODEL.DinoV3ZeroShot.BACKBONE, source='local', weights=self.cfg.MODEL.DinoV3ZeroShot.WEIGHT_PATH)
        self.model.eval()
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        self.model = self.model.to(self.device)

    def model_inference(self, input_data, domain):
        x_tokens_list = self.model.get_intermediate_layers(input_data, n=1, return_class_token=True)
        image_features = create_linear_input(x_tokens_list, 1, False)
        image_features = torch.nn.functional.normalize(image_features, dim=-1, eps=1e-6)
        return image_features
    
