from __future__ import annotations

import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ConvNextBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        weights = getattr(torchvision.models, "ConvNeXt_Tiny_Weights", None)
        model = torchvision.models.convnext_tiny(weights=weights.DEFAULT if weights else None)
        self.features = model.features
        self.out_channels = [192, 384, 768]

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in (3, 5, 7):
                feats.append(x)
        return feats

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        weights = getattr(torchvision.models, "ResNet50_Weights", None)
        model = torchvision.models.resnet50(weights=weights.DEFAULT if weights else None)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.out_channels = [512, 1024, 2048]

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]

class FPN(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.output = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels])
        self.p6 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, feats):
        laterals = [conv(feat) for conv, feat in zip(self.lateral, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        outs = [conv(feat) for conv, feat in zip(self.output, laterals)]
        p6 = self.p6(outs[-1])
        outs.append(p6)
        return outs

class FCOSHead(nn.Module):
    def __init__(self, channels: int = 256, num_classes: int = 5):
        super().__init__()
        self.cls_tower = self._make_tower(channels, dropout=0.1)
        self.box_tower = self._make_tower(channels, dropout=0.0)
        self.ctr_tower = self._make_tower(channels, dropout=0.1)
        self.cls_logits = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    @staticmethod
    def _make_tower(channels: int, dropout: float) -> nn.Sequential:
        layers = []
        for _ in range(4):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(32, channels),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    def forward(self, feats):
        cls, bbox, ctr = [], [], []
        for feat in feats:
            cls.append(self.cls_logits(self.cls_tower(feat)))
            bbox.append(F.relu(self.bbox_pred(self.box_tower(feat))))
            ctr.append(self.centerness(self.ctr_tower(feat)))
        return {"cls": cls, "bbox": bbox, "centerness": ctr}

class FCOSDetector(nn.Module):
    def __init__(self, num_classes: int = 5, backbone_name: str = "convnext_tiny"):
        super().__init__()
        try:
            self.backbone = ConvNextBackbone() if backbone_name == "convnext_tiny" else ResNetBackbone()
            self.backbone_name = backbone_name
        except Exception:
            self.backbone = ResNetBackbone()
            self.backbone_name = "resnet50"
        self.fpn = FPN(self.backbone.out_channels)
        self.head = FCOSHead(num_classes=num_classes)

    def forward(self, x):
        return self.head(self.fpn(self.backbone(x)))
