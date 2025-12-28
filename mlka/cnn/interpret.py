"""
mlka.cnn.interpret
------------------

Generic PyTorch interpretability utilities:
- Saliency maps
- Occlusion sensitivity
- Grad-CAM
- Dataset search

All functions are project-agnostic and reusable across ml-projects.
"""

from typing import Callable, Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"Expected shape (B,C,H,W) or (C,H,W), got {tuple(x.shape)}")
    return x


# =============================================================
# 1. SALIENCY MAPS
# =============================================================

def compute_saliency(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    model.eval()
    x = _ensure_batch(x)

    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)
    x.requires_grad_(True)

    scores = model(x)
    if target_class is None:
        target_class = scores.argmax(dim=1).item()

    score = scores[0, target_class]
    model.zero_grad()
    score.backward()

    grads = x.grad.detach()
    saliency = grads.abs().max(dim=1)[0].squeeze(0)

    saliency = saliency - saliency.min()
    denom = saliency.max()
    if denom > 0:
        saliency = saliency / denom

    return saliency.cpu()


def plot_saliency(
    image: torch.Tensor,
    saliency: torch.Tensor,
    title: str = "Saliency map",
    alpha: float = 0.6,
) -> None:
    img = image.detach().cpu().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[0] == 1:
        img = img[0]

    sal = saliency.detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(sal, cmap="jet", alpha=alpha)
    ax.axis("off")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# =============================================================
# 2. OCCLUSION SENSITIVITY
# =============================================================

def compute_occlusion_sensitivity(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    patch_size: int = 16,
    stride: int = 8,
    baseline: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    model.eval()
    x = _ensure_batch(x)

    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        base_scores = model(x)
        if target_class is None:
            target_class = base_scores.argmax(dim=1).item()
        base_score = base_scores[0, target_class].item()

    _, C, H, W = x.shape
    sensitivity = torch.zeros((H, W), device=device)

    baseline_patch = torch.full((C, patch_size, patch_size), baseline, device=device)

    for row in range(0, H, stride):
        for col in range(0, W, stride):
            r_end = min(row + patch_size, H)
            c_end = min(col + patch_size, W)

            x_occ = x.clone()
            patch_h = r_end - row
            patch_w = c_end - col

            x_occ[0, :, row:r_end, col:c_end] = baseline_patch[:, :patch_h, :patch_w]

            with torch.no_grad():
                scores = model(x_occ)
                drop = base_score - scores[0, target_class].item()

            sensitivity[row:r_end, col:c_end] += drop

    sensitivity = sensitivity - sensitivity.min()
    denom = sensitivity.max()
    if denom > 0:
        sensitivity = sensitivity / denom

    return sensitivity.cpu()


def plot_occlusion(
    image: torch.Tensor,
    sensitivity: torch.Tensor,
    title: str = "Occlusion sensitivity",
    alpha: float = 0.6,
) -> None:
    img = image.detach().cpu().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[0] == 1:
        img = img[0]

    sens = sensitivity.detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(sens, cmap="jet", alpha=alpha)
    ax.axis("off")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# =============================================================
# 3. GRAD-CAM
# =============================================================

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd(module, inp, out):
            # Save forward activations; detach later when building the CAM
            self.activations = out

        def bwd(module, grad_input, grad_output):
            # grad_output is a tuple; first element is gradient w.r.t. module output
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(fwd)
        # Use full backward hook when available (PyTorch 1.8+), fallback otherwise
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(bwd)
        else:
            self.target_layer.register_backward_hook(bwd)

    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:

        self.model.eval()
        x = _ensure_batch(x)
        if device is None:
            device = next(self.model.parameters()).device
        x = x.to(device)

        scores = self.model(x)
        if target_class is None:
            target_class = scores.argmax(dim=1).item()
        score = scores[0, target_class]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        A = self.activations
        G = self.gradients
        if A is None or G is None:
            # Fallback: return a zero heatmap if hooks did not fire (e.g., detached backbone)
            _, _, H, W = x.shape
            cam = torch.zeros((H, W), device=device)
            return cam.cpu()

        # Detach activations and gradients from the graph before further processing
        A = A.detach()
        G = G.detach()

        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam[0, 0]
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 0:
            cam = cam / denom

        _, _, H, W = x.shape
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        return cam.cpu()


def plot_gradcam(
    image: torch.Tensor,
    cam: torch.Tensor,
    title: str = "Grad-CAM",
    alpha: float = 0.6,
) -> None:
    img = image.detach().cpu().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[0] == 1:
        img = img[0]

    cam_np = cam.detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(cam_np, cmap="jet", alpha=alpha)
    ax.axis("off")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# =============================================================
# 4. UNIFIED INTERFACE
# =============================================================

def explain_prediction(
    model: torch.nn.Module,
    x: torch.Tensor,
    method: str = "saliency",
    target_class: Optional[int] = None,
    *,
    gradcam_layer: Optional[torch.nn.Module] = None,
    patch_size: int = 16,
    stride: int = 8,
    baseline: float = 0.0,
    device: Optional[torch.device] = None,
    plot: bool = True,
    title_prefix: str = "",
) -> torch.Tensor:

    method = method.lower()
    x_single = x if x.dim() == 3 else x[0]

    if method == "saliency":
        heatmap = compute_saliency(model, x, target_class, device)
        if plot:
            plot_saliency(x_single, heatmap, f"{title_prefix} Saliency")

    elif method == "occlusion":
        heatmap = compute_occlusion_sensitivity(
            model,
            x,
            target_class,
            patch_size,
            stride,
            baseline,
            device,
        )
        if plot:
            plot_occlusion(x_single, heatmap, f"{title_prefix} Occlusion")

    elif method == "gradcam":
        if gradcam_layer is None:
            raise ValueError("gradcam_layer is required for Grad-CAM")
        gc = GradCAM(model, gradcam_layer)
        heatmap = gc(x, target_class, device)
        if plot:
            plot_gradcam(x_single, heatmap, f"{title_prefix} Grad-CAM")
    else:
        raise ValueError(f"unknown method {method}")

    return heatmap


# =============================================================
# 5. DATASET SEARCH
# =============================================================

def search_dataset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    predicate: Callable[[Dict[str, Any]], bool],
    batch_size: int = 32,
    max_results: int = 32,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
) -> List[Dict[str, Any]]:

    if device is None:
        device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    results = []
    idx = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            for i in range(xb.size(0)):
                label = int(yb[i].item()) if torch.is_tensor(yb[i]) else int(yb[i])

                info = {
                    "index": idx,
                    "image": xb[i].cpu(),
                    "label": label,
                    "logits": logits[i].cpu(),
                    "probs": probs[i].cpu(),
                    "pred": int(preds[i].item()),
                    "correct": int(preds[i].item()) == label,
                }

                if predicate(info):
                    results.append(info)
                    if len(results) >= max_results:
                        return results
                idx += 1

    return results