import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet34
from preprocessing import preprocess_images_sequences


class To3Channels(nn.Module):

    def __init__(self,):
        super().__init__()
    def forward(self, img):
        return torch.cat([img, img, img], dim=0)

def main():
    # Base transforms for MoCo encoders
    transforms = T.Compose([
        T.ToTensor(),
        T.CenterCrop(224),
        To3Channels(),
    ])

    def transform_func(img):
        # Applies to all images at the same time
        img_range = [150, 350]
        img = (img - img_range[0])/(img_range[1]-img_range[0])
        img = img.astype(np.float32)
        img = transforms(img)
        return img


    model = resnet34(pretrained=True)  # Use a standard model 
    model.fc = nn.Identity()  # Remove final classification layer
    model.eval()
    model.cpu()  # Or .to("cuda:1") if available with local computer

    print(f"Encoder ready, model with {sum(p.numel() for p in model.parameters()):,} parameters")

    dataset_path =  r'F:\Data folder for ML\AU\AU'

    preprocess_images_sequences(model,
                                "r34p_10k_w6NewNEW",
                                transform_func,
                                device="cpu",#"cuda:1",
                                dataset_path=dataset_path)

if __name__ == "__main__":
    main()
