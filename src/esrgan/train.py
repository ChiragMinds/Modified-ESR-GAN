# src/esrgan/train.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from esrgan.data import get_loaders
from esrgan.models import Generator, Discriminator, VGG19PerceptualLoss
from esrgan.losses import psnr, ssim  # ensure losses.py exists as provided earlier

def save_model_and_weights(model, optimizer, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(save_path, f"model_checkpoint_epoch_{epoch}.pt"))
    torch.save(model.state_dict(), os.path.join(save_path, f"generator_epoch_{epoch}_liver.pt"))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True)
    p.add_argument('--val_dir', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--save_path', default='./saved_models')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--logdir', default='./runs')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    train_loader, val_loader = get_loaders(args.train_dir, args.val_dir, batch_size=args.batch_size)

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    perceptual = VGG19PerceptualLoss(device=device).to(device)

    adversarial_loss = nn.BCEWithLogitsLoss()
    content_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    gen_optimizer = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.9, 0.999))
    disc_optimizer = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.9, 0.999))

    writer = SummaryWriter(log_dir=os.path.join(args.logdir, 'esrgan'))

    # track metrics
    psnr_values = []
    ssim_values = []
    mse_values = []

    for epoch in range(1, args.epochs + 1):
        gen.train()
        disc.train()
        g_loss_sum = 0.0
        d_loss_sum = 0.0

        for low_res, high_res in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            batch_size = low_res.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # -----------------
            # Train generator
            # -----------------
            gen_optimizer.zero_grad()
            generated = gen(low_res)

            # content & perceptual losses — make sure ranges align (both in [0,1])
            L_content = content_loss(generated, high_res)
            L_percep = perceptual(generated, high_res)
            # adversarial term: discriminator should think generated is real
            adv_out = disc(generated)
            L_adv = adversarial_loss(adv_out, real_labels)

            gen_loss = L_content + 0.01 * L_percep + 1e-3 * L_adv  # weighted sum; tweak weights as needed
            gen_loss.backward()
            gen_optimizer.step()

            # -----------------
            # Train discriminator
            # -----------------
            disc_optimizer.zero_grad()
            real_out = disc(high_res)
            fake_out = disc(generated.detach())

            real_loss = adversarial_loss(real_out, real_labels)
            fake_loss = adversarial_loss(fake_out, fake_labels)
            disc_loss = (real_loss + fake_loss) * 0.5
            disc_loss.backward()
            disc_optimizer.step()

            g_loss_sum += gen_loss.item()
            d_loss_sum += disc_loss.item()

        avg_g_loss = g_loss_sum / len(train_loader)
        avg_d_loss = d_loss_sum / len(train_loader)

        # Validation
        gen.eval()
        val_psnr = 0.0
        val_mse = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for low_res, high_res in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                low_res = low_res.to(device)
                high_res = high_res.to(device)
                generated = gen(low_res)

                # Ensure inputs are in [0,1]; psnr implementation expects normalized range in losses.py
                val_psnr += psnr(generated, high_res).item()
                val_mse += mse_loss(generated, high_res).item()
                val_ssim += ssim(generated, high_res).item()

        avg_val_psnr = val_psnr / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)

        psnr_values.append(avg_val_psnr)
        mse_values.append(avg_val_mse)
        ssim_values.append(avg_val_ssim)

        # TensorBoard logs
        writer.add_scalar('Loss/Generator', avg_g_loss, epoch)
        writer.add_scalar('Loss/Discriminator', avg_d_loss, epoch)
        writer.add_scalar('PSNR/Validation', avg_val_psnr, epoch)
        writer.add_scalar('MSE/Validation', avg_val_mse, epoch)
        writer.add_scalar('SSIM/Validation', avg_val_ssim, epoch)

        print(f"Epoch {epoch}/{args.epochs} | G_loss: {avg_g_loss:.4f} | D_loss: {avg_d_loss:.4f} | PSNR: {avg_val_psnr:.4f} | SSIM: {avg_val_ssim:.4f}")

        # Save every epoch (or change to save every N epochs)
        save_model_and_weights(gen, gen_optimizer, epoch, args.save_path)

    writer.close()

    # After training, optionally save metrics to disk as numpy arrays
    np.save(os.path.join(args.save_path, "psnr_values.npy"), np.array(psnr_values))
    np.save(os.path.join(args.save_path, "ssim_values.npy"), np.array(ssim_values))
    np.save(os.path.join(args.save_path, "mse_values.npy"), np.array(mse_values))


if __name__ == '__main__':
    main()
