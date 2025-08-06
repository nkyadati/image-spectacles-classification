import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Map backbones to recommended input sizes
BACKBONE_IMG_SIZES = {
    "resnet50": 224,
    "mobilenetv3": 224,
    "efficientnet_b4": 380
}


class CustomClassifierWithConfidence(nn.Module):
    """
    A customizable classifier wrapper for multiple backbones (ResNet50, MobileNetV3, EfficientNet-B4)
    with softmax confidence outputs.

    Args:
        backbone (str): Backbone type ("resnet50", "mobilenetv3", "efficientnet_b4").
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pretrained weights.

    Raises:
        ValueError: If an unsupported backbone is provided.
    """
    def __init__(self, backbone="resnet50", num_classes=2, pretrained=True):
        super(CustomClassifierWithConfidence, self).__init__()
        self.backbone_name = backbone.lower()

        if self.backbone_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            self.fc = self.model.fc

        elif self.backbone_name == "mobilenetv3":
            self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_features, num_classes)
            self.fc = self.model.classifier[3]

        elif self.backbone_name == "efficientnet_b4":
            self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
            self.fc = self.model.classifier[1]

        else:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                             f"Choose from {list(BACKBONE_IMG_SIZES.keys())}.")

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            tuple:
                logits (torch.Tensor): Raw outputs (pre-softmax) of shape (B, num_classes).
                probs (torch.Tensor): Softmax probabilities of shape (B, num_classes).
        """
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


def get_model(backbone="resnet50", num_classes=2, pretrained=True):
    """
    Factory function to initialize a CustomClassifierWithConfidence model.

    Args:
        backbone (str): Backbone type ("resnet50", "mobilenetv3", "efficientnet_b4").
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        CustomClassifierWithConfidence: Initialized model instance.
    """
    return CustomClassifierWithConfidence(backbone=backbone, num_classes=num_classes, pretrained=pretrained)