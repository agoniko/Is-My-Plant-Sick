import torch
import sys

sys.path.append("../")
from src.Pretrained_models import TorchVit, Resnet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.helper_functions import HookContainerViT
import matplotlib.pyplot as plt
import cv2
import numpy as np

IMG_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = A.Compose(
    [
        A.Resize(*IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]
)

weights_path = "../weights/pretrained_vit.pt"


class AIEngineMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class AIEngine(metaclass=AIEngineMeta):
    def __init__(self):
        super().__init__()
        self.model = TorchVit(24)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        self.hook_container = HookContainerViT()
        self.hook_container.hook(self.model)

    def predict(self, img):
        image = transform(image=np.array(img))["image"]
        image = image.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(image)
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy().flatten()
            pred = np.argmax(prob)
            score = prob[pred]
            attention = self.hook_container.get_attentions(threshold=0.8, smooth=True)

        # normalizing img
        image = (np.array(img) / 255.0).astype(np.float32)

        # Apply the colormap to the attention map
        cmap_attention = plt.get_cmap("inferno")
        attention_colored = cmap_attention(attention)

        # Convert the attention map to an RGB format
        attention_colored_rgb = attention_colored[:, :, :3].astype(image.dtype)
        attention_colored_rgb = cv2.resize(
            attention_colored_rgb, (image.shape[1], image.shape[0])
        )

        # Blend the original image with the attention map
        blended_img = cv2.addWeighted(image, 0.5, attention_colored_rgb, 0.5, 0)

        return img, pred, score, blended_img
