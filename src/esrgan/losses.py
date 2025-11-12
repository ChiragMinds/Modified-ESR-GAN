# src/esrgan/losses.py
import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss

def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    # copy your ssim implementation, but avoid global device usage — use x.device
    K1 = 0.01
    K2 = 0.03
    L = 1.0  # use normalized range [0,1] if your input is in that range
    pad = window_size // 2

    x = x.view(-1, 1, x.shape[2], x.shape[3])
    y = y.view(-1, 1, y.shape[2], y.shape[3])

    window = torch.ones(1, 1, window_size, window_size, device=x.device) / window_size ** 2

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu_x = F.conv2d(x, window, padding=pad, groups=1)
    mu_y = F.conv2d(y, window, padding=pad, groups=1)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=1) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=1) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=1) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = SSIM_n / SSIM_d
    return ssim_map.mean() if size_average else ssim_map

def psnr(img1, img2):
    mse = mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(100.0)
    max_pixel = 1.0  # if normalized input [0,1]
    return 10 * torch.log10(max_pixel**2 / mse)
