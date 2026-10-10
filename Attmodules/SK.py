import torch
import torch.nn as nn
import torch.nn.functional as F


class SK(nn.Module):
    """
    SK (Selective Kernel) / SKConv
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    YOLOv5 兼容点：
    - YOLOv5 初始化会用 batch=1 做一次 forward 来推 stride
    - GAP 后是 (B, C, 1, 1)，此时 BatchNorm2d 在 train 模式下会报错
    - 所以 bottleneck 的 BN 必须去掉或改成 GroupNorm/LayerNorm
    """

    def __init__(self,c1: int,c2: int,kernels=(3, 5),reduction: int = 16,L: int = 32,groups: int = 1,act: str = "silu",):
        super().__init__()
        assert c1 > 0
        self.channels = c1
        self.kernels = list(kernels)
        self.M = len(self.kernels)

        # 激活函数
        if act.lower() == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.SiLU(inplace=True)

        # ---- 1) Split: 多分支卷积 ----
        self.branches = nn.ModuleList()
        for k in self.kernels:
            p = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(c1, c1, kernel_size=k, padding=p, groups=groups, bias=False),
                    nn.BatchNorm2d(c1),
                    self.act,
                )
            )

        # ---- 2) Fuse: GAP ----
        self.gap = nn.AdaptiveAvgPool2d(1)

        # bottleneck 维度 d
        d = max(c1 // reduction, L)

        # ✅ 关键修改：这里不要 BatchNorm2d(d)，否则 (1,d,1,1) 会报错
        self.fc = nn.Sequential(
            nn.Conv2d(c1, d, kernel_size=1, bias=False),
            self.act,
        )

        # 分支 logits
        self.fc_branches = nn.ModuleList([nn.Conv2d(d, c2, kernel_size=1, bias=True) for _ in range(self.M)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split
        feats = [br(x) for br in self.branches]          # list of (B,C,H,W)
        feats_stack = torch.stack(feats, dim=1)          # (B,M,C,H,W)

        # Fuse
        U = feats_stack.sum(dim=1)                       # (B,C,H,W)
        s = self.gap(U)                                  # (B,C,1,1)
        z = self.fc(s)                                   # (B,d,1,1)

        # Select
        logits = torch.stack([fc(z) for fc in self.fc_branches], dim=1)  # (B,M,C,1,1)
        attn = F.softmax(logits, dim=1)                                  # (B,M,C,1,1)

        # Weighted sum
        out = (feats_stack * attn).sum(dim=1)            # (B,C,H,W)
        return out
