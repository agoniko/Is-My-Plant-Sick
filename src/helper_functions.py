# a collection of helper functions
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from src.Vit import ViT
from src.Pretrained_models import TorchVit


def compare_models(models: list, test_loader, classes, device, suffixes: list = None):
    """
    Compare the performances of the given models on the test set.
    """
    results = {}
    assert suffixes is None or len(suffixes) == len(models)
    for i, model in tqdm(enumerate(models)):
        if suffixes is None:
            model_name = model.__class__.__name__
        else:
            model_name = model.__class__.__name__ + " " + suffixes[i]
        accuracy, precision, recall, f1, conf_mat = evaluate(
            model, test_loader, classes, device
        )
        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "conf_mat": conf_mat,
        }

    return results


def evaluate(model, test_loader, classes, device):
    model.eval()
    model = model.to(device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    conf_mat = confusion_matrix(y_true, y_pred, normalize="true")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return accuracy, precision, recall, f1, conf_mat


def gaussian_smooth_attention_map(attention_map, sigma=1):
    """
    Smoothens an attention map using Gaussian filtering.

    Parameters:
    - attention_map: torch.Tensor, the input attention map.
    - sigma: float, standard deviation of the Gaussian filter.

    Returns:
    - smoothed_attention_map: torch.Tensor, the smoothed attention map.
    """
    # Create a 2D Gaussian kernel
    kernel_size = int(6 * sigma + 1)  # Choose an appropriate kernel size
    kernel_size += 1 if kernel_size % 2 == 0 else 0
    gaussian_kernel = torch.from_numpy(
        np.exp(-((np.arange(kernel_size) - kernel_size // 2) ** 2) / (2 * sigma**2))
    ).float()
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # Apply 1D convolution along each dimension
    smoothed_attention_map = F.conv2d(
        attention_map.unsqueeze(0).unsqueeze(0),
        gaussian_kernel.view(1, 1, -1, 1),
        padding=kernel_size // 2,
    )
    smoothed_attention_map = F.conv2d(
        smoothed_attention_map,
        gaussian_kernel.view(1, 1, 1, -1),
        padding=kernel_size // 2,
    )

    w, h = attention_map.shape
    start = (kernel_size - 1) // 2
    end = start + w

    smoothed_attention_map = smoothed_attention_map[0, 0, start:end, start:end]

    return smoothed_attention_map


# Not used
def attention_rollout(As):
    """Computes attention rollout from the given list of attention matrices.
    https://arxiv.org/abs/2005.00928
    """

    # rollout = As[0]
    # for A in As[1:]:
    #    rollout = torch.matmul(
    #        0.5 * A + 0.5 * torch.eye(A.shape[1], device=A.device), rollout
    #    )  # the computation takes care of skip connections

    # compute the weighted mean of As increasing the weights as the index increases
    weight_step = 1 / len(As)
    rollout = As[0]
    for i, A in enumerate(As[1:]):
        rollout = torch.matmul(
            weight_step * (i + 1) * A
            + (1 - weight_step * (i + 1)) * torch.eye(A.shape[1], device=A.device),
            rollout,
        )  # the computation takes care of skip connections

    return rollout


def transform(img, img_size):
    img = transforms.Resize(img_size)(img)
    # img = transforms.ToTensor()(img)
    return img


def fit(
    model,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device = torch.device("cpu"),
    scheduler=None,
    save_path: str = "model.pt",
) -> (list, list):
    train_losses, valid_losses = [], []
    best_loss = np.inf

    epochs_range = tqdm(range(epochs))
    for epoch in epochs_range:
        model.train()
        running_train_loss = 0
        running_valid_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            out = model.forward(images)
            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            epochs_range.set_description(
                f"|Epoch: {epoch+1:^3} | {i / len(train_loader) * 100:.0f}%"
            )

        if scheduler is not None:
            scheduler.step()

        train_losses.append(running_train_loss / len(train_loader))
        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                out = model.forward(images)
                loss = criterion(out, labels)
                running_valid_loss += loss.item()

        valid_losses.append(running_valid_loss / len(valid_loader))
        epochs_range.write(
            f"|{'_'*11}| Train Loss: {train_losses[-1]:.3f} | Valid Loss: {valid_losses[-1]:.3f}"
        )
        if valid_losses[-1] < best_loss:
            best_loss = valid_losses[-1]
            torch.save(model.state_dict(), save_path)
    return train_losses, valid_losses


class HookContainerViT:
    def __init__(self):
        self.attentions = []
        self.average_methods = {"min": np.min, "mean": np.mean, "median": np.median}
        self.scale_factor = None

    def _hook_fn(self, module):
        # Capture the information you need and store it in the hook_container
        self.attentions.append(module.attention_weights.detach().cpu().clone())

    def hook(self, model):
        if isinstance(model, ViT):
            for layer in model.transformer_blocks[-1:]:
                layer.register_forward_hook(
                    lambda module, input, output: self._hook_fn(module)
                )
            self.scale_factor = 8
        elif isinstance(model, TorchVit):
            for layer in model.model.encoder.layers[-1:]:
                layer.register_forward_hook(
                    lambda module, input, output: self._hook_fn(module)
                )
            self.scale_factor = 16

    def get_attentions(
        self,
        average: str = "mean",
        threshold: float = 0.0,
        smooth: bool = False,
        sigma=7,
    ):
        """
        return a normalized [0, 1] attention map on which attention rollout is computed across the encoder blocks
        and the heads are averaged according to the average parameter
        average: str, either "min", "mean" or "median
        threshold: float [0, 1] threshold to apply to the attention map, set the percentage of lower values to be discarded
        smooth: bool, whether to apply a gaussian smoothing to the attention map for
        """
        attentions = self._process_attentions(average)
        self.attentions = []
        thr = np.percentile(attentions.flatten(), int(threshold * 100))
        attentions = np.where(attentions < thr, 0, attentions)
        if smooth:
            attentions = gaussian_smooth_attention_map(
                torch.tensor(attentions), sigma=sigma
            ).numpy()

        attentions = (attentions - attentions.min()) / (
            attentions.max() - attentions.min()
        )

        return attentions

    def _process_attentions(self, average: str = "mean"):
        attentions = torch.stack(self.attentions).squeeze(1)

        # attention rollout is not performed, only the last layer is taken
        attentions = attention_rollout(attentions)

        nh = attentions.shape[0]  # number of head
        # keep only the output patch attention (class patch)
        attentions = attentions[:, 0, 1:]
        # since we are dealing with squared images we can take the square root of the last dimension as the width and height
        width = height = int(np.sqrt(attentions.shape[-1]))

        attentions = attentions.reshape(nh, width, height)
        attentions = (
            nn.functional.interpolate(
                attentions.unsqueeze(0), scale_factor=self.scale_factor, mode="nearest"
            )[0]
            .detach()
            .cpu()
            .numpy()
        )

        if average not in self.average_methods:
            raise ValueError(
                f"average must be one of {list(self.average_methods.keys())}"
            )

        attentions = self.average_methods[average](attentions, axis=0)


        return attentions


class HookContainerSwin:
    def __init__(self):
        self.attentions = None

    def _hook_fn(self, module):
        # Capture the information you need and store it in the hook_container
        self.attentions = module.attention.detach().cpu().clone()
        # hook_container.value = output.clone()

    def hook(self, model):
        # for layer in self.target_layers:
        #    model.backbone[0][layer][-1].attn.register_forward_hook(
        #        lambda module, input, output: self._hook_fn(module)
        #    )
        model.backbone[0][7][-1].attn.register_forward_hook(
            lambda module, input, output: self._hook_fn(module)
        )

    def get_attentions(self, threshold: float = 0.0, smooth: bool = False):
        """
        return a normalized [0, 1] attention map that is simply the mean of the attention maps of the heads of the last layer of the swin model
        and the heads are averaged according to the average parameter
        threshold: float [0, 1] threshold to apply to the attention map, set the percentage of lower values to be discarded
        smooth: bool, whether to apply a gaussian smoothing to the attention map for
        """
        attentions = self._process_attentions()
        self.attentions = []
        thr = np.percentile(attentions.flatten(), int(threshold * 100))
        attentions = np.where(attentions < thr, 0, attentions)
        if smooth:
            attentions = gaussian_smooth_attention_map(
                torch.tensor(attentions), sigma=10
            ).numpy()

        return attentions

    def _process_attentions(self):
        # we average the results across the first 3 dimensions
        # I know that this is not the best way since, e.g. for the last two dimensions I have the self attention matrix
        # I haven't found a documentation about how to visualize Swin attention maps
        mean_attn = torch.mean((self.attentions), dim=(1, 2))
        # mean_attn = torch.stack(mean_attn)
        mean_attn = torch.mean(mean_attn, dim=0)
        # mean_attn = torch.stack(mean_attn)
        # mean_attn = torch.mean(mean_attn, dim=0)
        mean_attn = mean_attn.reshape(1, 7, 7)

        # normalize the attention map
        mean_attn = (mean_attn - mean_attn.min()) / (mean_attn.max() - mean_attn.min())
        attentions = (
            nn.functional.interpolate(
                mean_attn.unsqueeze(0), scale_factor=32, mode="nearest"
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        # attentions = attentions.mean(axis=0)

        # normalize the attention map
        attentions = (attentions - attentions.min()) / (
            attentions.max() - attentions.min()
        )
        return attentions[0]


# class HookContainerSwin:
#    def __init__(self):
#        self.attentions = []
#        self.target_layers = [1, 3, 5, 7]
#
#    def _hook_fn(self, module):
#        # Capture the information you need and store it in the hook_container
#        self.attentions.append(module.attention.detach().cpu().clone())
#        # hook_container.value = output.clone()
#
#    def hook(self, model):
#        for layer in self.target_layers:
#            model.backbone[0][layer][-1].attn.register_forward_hook(
#                lambda module, input, output: self._hook_fn(module)
#            )
#
#    def get_attentions(self, threshold: float = 0.0, smooth: bool = False):
#        """
#        return a normalized [0, 1] attention map that is simply the mean of the attention maps of the heads of the last layer of the swin model
#        and the heads are averaged according to the average parameter
#        threshold: float [0, 1] threshold to apply to the attention map, set the percentage of lower values to be discarded
#        smooth: bool, whether to apply a gaussian smoothing to the attention map for
#        """
#        attentions = self._process_attentions()
#        self.attentions = []
#        thr = np.percentile(attentions.flatten(), int(threshold * 100))
#        attentions = np.where(attentions < thr, 0, attentions)
#        if smooth:
#            attentions = gaussian_smooth_attention_map(
#                torch.tensor(attentions), sigma=10
#            ).numpy()
#
#        return attentions
#
#    def _process_attentions(self):
#        # we average the results across the first 3 dimensions
#        # I know that this is not the best way since, e.g. for the last two dimensions I have the self attention matrix
#        # I haven't found a documentation about how to visualize Swin attention maps
#
#        mean_attn = [torch.mean((attn), dim=(0, 1, 2)) for attn in self.attentions]
#        mean_attn = torch.stack(mean_attn)
#        mean_attn = torch.mean(mean_attn, dim=0)
#        #mean_attn = torch.stack(mean_attn)
#        # mean_attn = torch.mean(mean_attn, dim=0)
#        mean_attn = mean_attn.reshape(1, 7, 7)
#
#        # normalize the attention map
#        mean_attn = (mean_attn - mean_attn.min()) / (mean_attn.max() - mean_attn.min())
#        attentions = (
#            nn.functional.interpolate(
#                mean_attn.unsqueeze(0), scale_factor=32, mode="nearest"
#            )[0]
#            .detach()
#            .cpu()
#            .numpy()
#        )
#        #attentions = attentions.mean(axis=0)
#
#        # normalize the attention map
#        attentions = (attentions - attentions.min()) / (
#            attentions.max() - attentions.min()
#        )
#        return attentions[0]
