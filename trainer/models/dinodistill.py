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

class DinoStudent(nn.Module):
    def __init__(
        self,
        cfg,
        student_backbone,
        dimention_student,
        dimention_teacher,
        num_classes_per_domain,
    ):
        super().__init__()
        self.backbone = student_backbone
        self.cfg = cfg
        self.proj = nn.Linear(dimention_student, dimention_teacher, bias=False)
        self.classifiers = nn.ModuleDict(
            {
                f"domain_{int(domain)}": nn.Linear(
                    dimention_student, int(num_classes)
                )
                for domain, num_classes in num_classes_per_domain.items()
            }
        )

    def forward(self, x):
        feature_student = create_linear_input(
            self.backbone.get_intermediate_layers(x, n=1, return_class_token=True),
            1,
            False,
        )
        feature_student_teacher_space = self.proj(feature_student)
        return feature_student_teacher_space, feature_student

    def classifier_logits(self, features, domains):
        if len(domains) == 0:
            raise ValueError("Expected at least one domain entry to compute logits.")
        domain = domains[0]
        if isinstance(domain, torch.Tensor):
            domain = domain.item()
        for dom in domains:
            dom_val = dom.item() if isinstance(dom, torch.Tensor) else dom
            if dom_val != domain:
                raise ValueError(
                    "Mixed-domain batches are not supported for DinoStudent classifiers."
                )
        classifier_key = f"domain_{int(domain)}"
        if classifier_key not in self.classifiers:
            raise KeyError(f"No classifier registered for {classifier_key}.")
        return self.classifiers[classifier_key](features)

@MODEL_REGISTRY.register()
class DinoDistill(Trainer):
    """
    Distillation from DINO teacher to DINO student with optional finetuning on ReID task.
    """

    def build_model(self):
        print("Loading Dino backbone: {}".format(self.cfg.MODEL.DinoDistill.BACKBONE))
        dino_teacher = torch.hub.load(self.cfg.MODEL.DinoDistill.REPO, self.cfg.MODEL.DinoDistill.BACKBONE, source='local', 
                                    weights=self.cfg.MODEL.DinoDistill.WEIGHT_PATH)
        self.dino_teacher = dino_teacher.to(self.device)
        self.dino_teacher.eval()
        for param in self.dino_teacher.parameters():
            param.requires_grad_(False)

        student_backbone = torch.hub.load(
            self.cfg.MODEL.DinoDistill.REPO,
            self.cfg.MODEL.DinoDistill.STUDENT_BACKBONE,
            source="local",
            weights=self.cfg.MODEL.DinoDistill.STUDENT_WEIGHT_PATH,
        )

        teacher_embed_dim = getattr(self.dino_teacher, "embed_dim", None) or getattr(
            self.dino_teacher, "num_features", None
        )
        student_embed_dim = getattr(student_backbone, "embed_dim", None) or getattr(
            student_backbone, "num_features", None
        )
        if teacher_embed_dim is None or student_embed_dim is None:
            raise AttributeError(
                "Unable to determine embed dimensions for teacher/student backbones required for projection."
            )

        self.dino_student = DinoStudent(
            self.cfg,
            student_backbone,
            student_embed_dim,
            teacher_embed_dim,
            self.num_classes,
        ).to(self.device)

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
        # print("Parameters to be updated: {}".format(enabled_params))

        self.dino_student.to(self.device)


    def forward_backward(self, batch_data):
        image, target, domains, time = self.parse_batch_train(batch_data)
        domain = domains[0]  # Assume all samples in batch are from the same domain for training
        domain = domain.item() if isinstance(domain, torch.Tensor) else domain

        student_teacher_space, student_native = self.dino_student(image)
        student_teacher_space = nn.functional.normalize(
            student_teacher_space, dim=-1, eps=1e-6
        )

        distillation_loss = torch.tensor(0.0, device=image.device)
        reid_loss = torch.tensor(0.0, device=image.device)
        id_loss = torch.tensor(0.0, device=image.device)
        tri_loss = torch.tensor(0.0, device=image.device)
        if self.cfg.MODEL.DinoDistill.DISTILL:
            with torch.no_grad():
                teacher_output = create_linear_input(self.dino_teacher.get_intermediate_layers(image, n=1, return_class_token=True), 1, False)
                teacher_output = nn.functional.normalize(teacher_output, dim=-1, eps=1e-6)

            distillation_loss = F.kl_div(
                F.log_softmax(student_teacher_space / 0.2, dim=1),
                F.softmax(teacher_output / 0.2, dim=1),
                reduction="batchmean",
            ) * 0.2**2

        if self.cfg.MODEL.DinoDistill.FINETUNE:
            loss_func, center_criterion = make_loss(self.cfg, self.num_classes[domain], self.device)
            cls_scores = self.dino_student.classifier_logits(student_native, domains)
            student_native_normalized = nn.functional.normalize(
                student_native, dim=-1, eps=1e-6
            )
            loss, ID_LOSS, TRI_LOSS = loss_func(
                cls_scores, student_native_normalized, target, None
            )
            reid_loss = loss
            id_loss = (
                ID_LOSS
                if torch.is_tensor(ID_LOSS)
                else torch.as_tensor(ID_LOSS, device=image.device)
            )
            tri_loss = (
                TRI_LOSS
                if torch.is_tensor(TRI_LOSS)
                else torch.as_tensor(TRI_LOSS, device=image.device)
            )
        loss = distillation_loss + reid_loss

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "distillation_loss": distillation_loss.item(),
            "reid_loss": reid_loss.item(),
            "ID_LOSS": id_loss.item(),
            "TRI_LOSS": tri_loss.item(),
        }

        return loss_summary

    def model_inference(self, batch_data, domains):
        _, feat = self.dino_student(batch_data)
        feat = nn.functional.normalize(feat, dim=-1, eps=1e-6)
        return feat
    
