import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        if x.dtype != self.conv.weight.dtype:
            x = x.to(self.conv.weight.dtype)
        return self.act(self.bn(self.conv(x)))


class FeatureDistillLoss(nn.Module):
    def __init__(self, loss_type='mse', normalize=False, temperature=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.temperature = temperature

    def forward(self, student_feats, teacher_feats, weights=None):
        if not student_feats or not teacher_feats:
            return torch.tensor(0.0, device=student_feats[0].device if student_feats else 'cpu')

        assert len(student_feats) == len(teacher_feats)
        total_loss = 0.0
        n_layers = len(student_feats)

        if weights is None:
            weights = [1.0 / n_layers] * n_layers

        for i, (s_feat, t_feat) in enumerate(zip(student_feats, teacher_feats)):
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:],
                                       mode='bilinear', align_corners=False)

            s_feat = s_feat / self.temperature
            t_feat = t_feat / self.temperature

            if self.loss_type == 'mse':
                loss = F.mse_loss(s_feat, t_feat)
            elif self.loss_type == 'l1':
                loss = F.l1_loss(s_feat, t_feat)
            elif self.loss_type == 'smooth_l1':
                loss = F.smooth_l1_loss(s_feat, t_feat)
            elif self.loss_type == 'cosine':
                s_norm = F.normalize(s_feat.view(s_feat.size(0), -1), dim=1)
                t_norm = F.normalize(t_feat.view(t_feat.size(0), -1), dim=1)
                loss = 1 - (s_norm * t_norm).sum(dim=1).mean()
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            if self.normalize:
                t_norm = torch.norm(t_feat, p=2) + 1e-8
                loss = loss / t_norm

            total_loss += weights[i] * loss

        return total_loss


class FeatureDistillManager(nn.Module):
    def __init__(self, teacher_model, student_model,
                 distill_layers=['model.17.cv2', 'model.20.cv2', 'model.23.cv2'],
                 adapter_channels=None):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distill_layers = distill_layers
        self._handles = []

        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

        self.adapters = nn.ModuleList()
        if adapter_channels:
            for in_ch, out_ch in adapter_channels:
                self.adapters.append(FeatureAdapter(in_ch, out_ch))

        self.teacher_features = OrderedDict()
        self.student_features = OrderedDict()
        self._register_hooks()

    def _register_hooks(self):
        for name in self.distill_layers:
            for n, m in self.teacher_model.named_modules():
                if n == name:
                    handle = m.register_forward_hook(self._get_hook(name, 'teacher'))
                    self._handles.append(handle)
            for n, m in self.student_model.named_modules():
                if n == name:
                    handle = m.register_forward_hook(self._get_hook(name, 'student'))
                    self._handles.append(handle)

    def _get_hook(self, name, model_type):
        def hook(module, input, output):
            if model_type == 'teacher':
                self.teacher_features[name] = output.detach()
            else:
                self.student_features[name] = output
        return hook

    def get_distill_loss(self, loss_fn, weights=None):
        teacher_feats = [self.teacher_features.get(l, None) for l in self.distill_layers]
        student_feats = [self.student_features.get(l, None) for l in self.distill_layers]

        teacher_feats = [f for f in teacher_feats if f is not None]
        student_feats = [f for f in student_feats if f is not None]

        if not teacher_feats or not student_feats:
            return torch.tensor(0.0, device=next(self.teacher_model.parameters()).device)

        if len(self.adapters) > 0:
            adapted_feats = []
            for i, feat in enumerate(student_feats):
                if i < len(self.adapters):
                    adapted_feats.append(self.adapters[i](feat))
                else:
                    adapted_feats.append(feat)
            student_feats = adapted_feats

        return loss_fn(student_feats, teacher_feats, weights)