import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


class SwinV2TwinRGBD(nn.Module):
    def __init__(self, classifier):
        super().__init__()

        self.net_rgb = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-base-patch4-window12-192-22k",
            ignore_mismatched_sizes=True,
            output_hidden_states=True,
            output_attentions=True,
        )
        self.net_depth = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-base-patch4-window12-192-22k",
            ignore_mismatched_sizes=True,
            output_hidden_states=True,
            output_attentions=True,
        )
        self.net_rgb.classifier = nn.Identity()
        self.net_depth.classifier = nn.Identity()
        self.classifier = classifier  # nn.Linear(in_features=1024*2, out_features=102, bias=True)
        self.hook_output_rgb = None
        self.hook_output_depth = None
        self.net_rgb.swinv2.pooler.register_forward_hook(self.hook_fn_rgb)
        self.net_depth.swinv2.pooler.register_forward_hook(self.hook_fn_depth)

    def hook_fn_rgb(self, module, input, output):
        self.hook_output_rgb = output

    def hook_fn_depth(self, module, input, output):
        self.hook_output_depth = output

    def forward(self, pixel_values_rgb, pixel_values_depth):
        _ = self.net_rgb(pixel_values=pixel_values_rgb)
        _ = self.net_depth(pixel_values=pixel_values_depth)
        features_rgb = self.hook_output_rgb.squeeze(-1)
        features_depth = self.hook_output_depth.squeeze(-1)
        x = torch.cat((features_rgb, features_depth), dim=1)
        logits = self.classifier(x)
        return {
            "logits": logits,
            "features_rgb": features_rgb,
            "features_depth": features_depth,
        }
