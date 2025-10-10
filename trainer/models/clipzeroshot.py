import os

import torch
from clip import clip

from trainer import MODEL_REGISTRY, Trainer
from utils import PROMPT_TEMPLATES


@MODEL_REGISTRY.register()
class CLIPZeroShot(Trainer):
    def build_model(self):
        self.clip_model, _ = clip.load(
            self.cfg.MODEL.CLIPZeroShot.BACKBONE,
            device=self.device,
            download_root=os.path.abspath(os.path.expanduser("data")),
        )

    def model_inference(self, image, domain):
        image_features = self.clip_model.encode_image(image)
        image_features = torch.nn.functional.normalize(image_features, dim=-1, eps=1e-6)
        return image_features
