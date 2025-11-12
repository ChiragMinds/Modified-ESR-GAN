# src/esrgan/train.py
import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from esrgan.data import get_loaders
from esrgan.models import Generator, Discriminator, VGG19PerceptualLoss
from esrgan.losses import psnr, ssim
import torch.optim as optim
import torch.nn as nn

def save_model_and_weights(model, optimizer, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{save_path}/model_checkpoint_epoch_{epoch}.pt")
    torch.save(model.state_dict(), f"{save_path}/generator_epoch_{epoch}_liver.pt")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True)
    p.add_argument('--val_dir', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--save_path', default='./saved_models')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    train_loader, val_loader = get_loaders(args.train_dir, args.val_dir, batch_size=args.batch_size)
    gen = Generator().to(device)
    disc = Discriminator().to(device)
    perceptual = VGG19PerceptualLoss().to(device)

    adversarial_loss = nn.BCEWithLogitsLoss()
    content_loss = nn.L1Loss()
    gen_optim = optim.Adam(gen.parameters(), lr=args.lr)
    disc_optim = optim.Adam(disc.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=os.path.join(args.save_path, 'tensorboard'))

    # Training loop
    for epoch in range(1, n_epochs + 1):
        gen.train()
        disc.train()
        g_loss_sum = 0
        d_loss_sum = 0
    
        for low_res, high_res in tqdm(train_loader):
            low_res, high_res = low_res.to(device), high_res.to(device)
            real_labels = torch.ones(low_res.size(0), 1).to(device)
            fake_labels = torch.zeros(low_res.size(0), 1).to(device)
    
            # Train the generator
            gen_optimizer.zero_grad()
            generated_images = gen(low_res)
            gen_content_loss = content_loss(generated_images, high_res)
            gen_perceptual_loss = perceptual_loss(generated_images, high_res)
            gen_loss = gen_content_loss + gen_perceptual_loss + adversarial_loss(disc(generated_images), real_labels)
            gen_loss.backward()
            gen_optimizer.step()
    
            # Train the discriminator
            disc_optimizer.zero_grad()
            real_loss = adversarial_loss(disc(high_res), real_labels)
            fake_loss = adversarial_loss(disc(generated_images.detach()), fake_labels)
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            disc_optimizer.step()
    
            g_loss_sum += gen_loss.item()
            d_loss_sum += disc_loss.item()
    
        avg_g_loss = g_loss_sum / len(train_loader)
        avg_d_loss = d_loss_sum / len(train_loader)
    
        # Validation
        gen.eval()
        val_psnr = 0
        val_mse = 0
        val_ssim = 0
        with torch.no_grad():
            for low_res, high_res in tqdm(val_loader):
                low_res, high_res = low_res.to(device), high_res.to(device)
                generated_images = gen(low_res)
                val_psnr += psnr(generated_images, high_res).item()
                val_mse += mse_loss_func(generated_images, high_res).item()
                val_ssim += ssim(generated_images, high_res).item()
    
        avg_val_psnr = val_psnr / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
    
        # Store the values for plotting
        psnr_values.append(avg_val_psnr)
        ssim_values.append(avg_val_ssim)
        mse_values.append(avg_val_mse)
    
        # TensorBoard logging
        writer.add_scalar('Loss/Generator', avg_g_loss, epoch)
        writer.add_scalar('Loss/Discriminator', avg_d_loss, epoch)
        writer.add_scalar('PSNR/Validation', avg_val_psnr, epoch)
        writer.add_scalar('MSE/Validation', avg_val_mse, epoch)
        writer.add_scalar('SSIM/Validation', avg_val_ssim, epoch)
    
        print(f"Epoch {epoch}/{n_epochs}, Generator Loss: {round(avg_g_loss,2)}, Discriminator Loss: {round(avg_d_loss,2)}")
        print(f"Validation PSNR: {round(avg_val_psnr,2)}, MSE: {round(avg_val_mse,2)}, SSIM: {round(avg_val_ssim,2)}")
    
        # Save the model
        save_model_and_weights(gen, gen_optimizer, epoch, save_path)

    writer.close()

if __name__ == '__main__':
    main()
