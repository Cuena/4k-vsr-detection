import torch
import torch.nn as nn
import torchvision.models as models
from contextlib import nullcontext

class ConvnextFeatureExtractor(nn.Module):
    def __init__(self, 
                    model_path: str = None, 
                    freeze: bool = False, 
                ):
        super().__init__()

        self.convnext_model = models.convnext_base(weights=None)
        if model_path:
            self.convnext_model.load_state_dict(torch.load(model_path))
        else:
            print("No model path provided, attempting to download pre-trained model.")
            self.convnext_model = models.convnext_base(pretrained=True)

        self.convnext_model = nn.Sequential(*list(self.convnext_model.children())[:-1])
        
        self.freeze = freeze
        if self.freeze:
            self.convnext_model.eval()

    def forward(self, x):
        with torch.no_grad() if self.freeze else nullcontext():
            return self.convnext_model(x).view(x.size(0), -1)

if __name__ == "__main__":
    _ = ConvnextFeatureExtractor(None, False)