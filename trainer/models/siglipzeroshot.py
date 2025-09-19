import argparse
import logging
import os
import torch
import torch.nn as nn

from transformers import AutoModel, AutoProcessor, SiglipVisionModel
from trainer import MODEL_REGISTRY, Trainer

from utils import PROMPT_TEMPLATES


@MODEL_REGISTRY.register()
class SIGLIPZeroShot(Trainer):
    def build_model(self):
        ckpt = self.cfg.MODEL.SIGLIPZeroShot.CKPT
        self.siglip_model = AutoModel.from_pretrained(ckpt)
        
        # Move model to the appropriate device (GPU if available)
        self.siglip_model = self.siglip_model.to(self.device)

    def model_inference(self, input_data):
        image_features = self.siglip_model.get_image_features(**{'pixel_values': input_data})
        image_features = torch.nn.functional.normalize(image_features, dim=-1, eps=1e-6)
        return image_features