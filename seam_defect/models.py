"""Seam classifiers shared by training and inference."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-06):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        p = torch.clamp(self.p, min=1.0, max=6.0)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / p).flatten(1)


class FPNHead(nn.Module):

    def __init__(self, input_channels, num_classes=11, fpn_channels=256, dropout=0.3):
        super().__init__()

        def lateral_block(in_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, fpn_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, fpn_channels),
                nn.ReLU(inplace=True),
            )

        def smooth_block():
            return nn.Sequential(
                nn.Conv2d(
                    fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.GroupNorm(32, fpn_channels),
                nn.ReLU(inplace=True),
            )

        self.lateral = nn.ModuleList([lateral_block(c) for c in input_channels])
        self.smooth = nn.ModuleList([smooth_block() for _ in range(4)])
        self.dropout2d = nn.Dropout2d(0.1)
        self.gem = GeM()
        self.classifier = nn.Sequential(
            nn.Linear(fpn_channels * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, c2, c3, c4, c5):
        features = [c2, c3, c4, c5]
        l2, l3, l4, l5 = [self.lateral[i](features[i]) for i in range(4)]
        p5 = l5
        p4 = l4 + F.interpolate(p5, size=l4.shape[-2:], mode="nearest")
        p3 = l3 + F.interpolate(p4, size=l3.shape[-2:], mode="nearest")
        p2 = l2 + F.interpolate(p3, size=l2.shape[-2:], mode="nearest")
        pyramid = [
            self.smooth[0](p2),
            self.smooth[1](p3),
            self.smooth[2](p4),
            self.smooth[3](p5),
        ]
        pooled = [self.gem(self.dropout2d(feature)) for feature in pyramid]
        combined = torch.cat(pooled, dim=1)
        return self.classifier(combined)


class FPNClassifier(nn.Module):

    def __init__(self, architecture, num_classes=11, pretrained=True, dropout=0.3):
        super().__init__()
        self.architecture = architecture
        if architecture == "densenet121_fpn":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            backbone = models.densenet121(weights=weights)
            backbone.classifier = nn.Identity()
            channels = [256, 512, 1024, 1024]
            self.backbone_type = "densenet121"
        elif architecture == "convnext_tiny_fpn":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            backbone.classifier = nn.Identity()
            channels = [96, 192, 384, 768]
            self.backbone_type = "convnext_tiny"
        elif architecture == "efficientnet_b0_fpn":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            backbone.classifier = nn.Identity()
            channels = [24, 40, 112, 1280]
            self.backbone_type = "efficientnet_b0"
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.backbone = backbone
        self.fpn_head = FPNHead(
            input_channels=channels,
            num_classes=num_classes,
            fpn_channels=256,
            dropout=dropout,
        )

    def extract_features(self, x):
        f = self.backbone.features
        if self.backbone_type == "densenet121":
            x = f.conv0(x)
            x = f.norm0(x)
            x = f.relu0(x)
            x = f.pool0(x)
            c2 = f.denseblock1(x)
            x = f.transition1(c2)
            c3 = f.denseblock2(x)
            x = f.transition2(c3)
            c4 = f.denseblock3(x)
            x = f.transition3(c4)
            c5 = f.denseblock4(x)
            c5 = F.relu(f.norm5(c5), inplace=False)
        elif self.backbone_type == "convnext_tiny":
            x = f[0](x)
            c2 = f[1](x)
            x = f[2](c2)
            c3 = f[3](x)
            x = f[4](c3)
            c4 = f[5](x)
            x = f[6](c4)
            c5 = f[7](x)
        elif self.backbone_type == "efficientnet_b0":
            x = f[0](x)
            x = f[1](x)
            c2 = f[2](x)
            c3 = f[3](c2)
            x = f[4](c3)
            c4 = f[5](x)
            x = f[6](c4)
            x = f[7](x)
            c5 = f[8](x)
        return (c2, c3, c4, c5)

    def forward(self, x):
        features = self.extract_features(x)
        return self.fpn_head(*features)


class ResNet18FPN(nn.Module):

    def __init__(self, num_classes=11, fpn_channels=128, dropout=0.3, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.lateral2 = nn.Conv2d(64, fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(128, fpn_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(256, fpn_channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(512, fpn_channels, kernel_size=1)
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth5 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.classifier = nn.Sequential(
            nn.Linear(fpn_channels * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3)
        p2 = self.lateral2(c2)
        p4 = p4 + nn.functional.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p3 = p3 + nn.functional.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p2 = p2 + nn.functional.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        f2 = nn.functional.adaptive_avg_pool2d(p2, output_size=1).flatten(1)
        f3 = nn.functional.adaptive_avg_pool2d(p3, output_size=1).flatten(1)
        f4 = nn.functional.adaptive_avg_pool2d(p4, output_size=1).flatten(1)
        f5 = nn.functional.adaptive_avg_pool2d(p5, output_size=1).flatten(1)
        fused = torch.cat([f2, f3, f4, f5], dim=1)
        output = self.classifier(fused)
        return output
