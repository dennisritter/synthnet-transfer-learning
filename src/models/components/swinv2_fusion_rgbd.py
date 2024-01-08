import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


class SwinV2FusionRGBD(nn.Module):
    def __init__(self, classifier):
        super().__init__()

        self.net = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-base-patch4-window12-192-22k",
            ignore_mismatched_sizes=True,
            output_hidden_states=True,
            output_attentions=True,
        )
        self.net.classifier = nn.Identity()
        self.classifier = classifier

    def forward(self, pixel_values_rgb, pixel_values_depth):
        out_rgb = self.net(pixel_values=pixel_values_rgb)
        out_depth = self.net(pixel_values=pixel_values_depth)
        features_rgb = out_rgb["hidden_states"][-1][:, 0, :]
        features_depth = out_depth["hidden_states"][-1][:, 0, :]
        x = torch.cat((features_rgb, features_depth), dim=1)
        logits = self.classifier(x)
        return {
            "logits": logits,
            "features_rgb": features_rgb,
            "features_depth": features_depth,
        }
