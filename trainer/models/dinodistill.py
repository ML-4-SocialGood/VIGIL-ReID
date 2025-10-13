import os

import torch
import torch.nn as nn
from .dino_utils import create_linear_input
from torch.nn import functional as F
from metrics import compute_accuracy
from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from utils import PROMPT_TEMPLATES
from loss.make_loss import make_loss

@MODEL_REGISTRY.register()
class DinoDistill(Trainer):
    """
    Dino-Adapter for multi-source domain adaptation
    """

    def build_model(self):
        print("Loading Dino backbone: {}".format(self.cfg.MODEL.DinoDistill.BACKBONE))
        dino_teacher = torch.hub.load(self.cfg.MODEL.DinoDistill.REPO, self.cfg.MODEL.DinoDistill.BACKBONE, source='local', 
                                    weights=self.cfg.MODEL.DinoDistill.WEIGHT_PATH)
        self.dino_teacher = dino_teacher.to(self.device)
        self.dino_teacher.eval()
        for param in self.dino_teacher.parameters():
            param.requires_grad_(False)

        dino_student = torch.hub.load(self.cfg.MODEL.DinoDistill.REPO, self.cfg.MODEL.DinoDistill.STUDENT_BACKBONE, source='local', 
                        weights=self.cfg.MODEL.DinoDistill.STUDENT_WEIGHT_PATH)
        self.dino_student = dino_student.to(self.device)
        self.optimizer = build_optimizer(self.dino_student, self.cfg.OPTIM)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)


        self.dino_student.train()
        self.model_registeration(
            "dino_student",
            self.dino_student,
            self.optimizer,
            self.lr_scheduler,
        )

        # Double check
        enabled_params = set()
        for name, param in self.dino_student.named_parameters():
            if param.requires_grad:
                enabled_params.add(name)
        print("Parameters to be updated: {}".format(enabled_params))

        self.dino_student.to(self.device)


    def forward_backward(self, batch_data):
        image, target, domains, time = self.parse_batch_train(batch_data)
        domain = domains[0]  # Assume all samples in batch are from the same domain for training

        student_output = create_linear_input(self.dino_student.get_intermediate_layers(image, n=1, return_class_token=True), 1, False)
        student_output = nn.functional.normalize(student_output, dim=-1, eps=1e-6)

        distillation_loss = 0
        reid_loss = 0
        if self.cfg.MODEL.DinoDistill.DISTILL:
            with torch.no_grad():
                teacher_output = create_linear_input(self.dino_teacher.get_intermediate_layers(image, n=1, return_class_token=True), 1, False)
                teacher_output = nn.functional.normalize(teacher_output, dim=-1, eps=1e-6)

            distillation_loss = F.kl_div(F.log_softmax(student_output / 0.2, dim=1), F.softmax(teacher_output / 0.2, dim=1), reduction="batchmean", ) * 0.2**2

        if self.cfg.MODEL.DinoDistill.FINETUNE:
            loss_func, center_criterion = make_loss(self.cfg, self.num_classes[domain], self.device)
            loss, ID_LOSS, TRI_LOSS = loss_func(student_output, student_output, target, None)
            reid_loss = loss
        loss = distillation_loss + reid_loss

        # Use domain-specific optimizer for backward and update
        self.model_backward_and_update(loss)

        loss_summary = {
            "domain": self.data_manager.get_source_domains[domain],
            "domain_id": domain,
            "loss": loss.item(),
            "distillation_loss": distillation_loss.item(),
            "reid_loss": reid_loss.item(),
        }

        # LR scheduler now steps once per epoch in Trainer.after_epoch

        return loss_summary

    def model_inference(self, batch_data, domains, time):
        _, feat = self.dino_student(batch_data, domains, time)
        return feat
    
