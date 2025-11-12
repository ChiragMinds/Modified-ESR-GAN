# src/esrgan/test.py
import argparse
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from esrgan.models import Generator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='generated.png')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    gen = Generator().to(device)
    state = torch.load(args.model_path, map_location=device)
    # if saved as state_dict:
    if 'model_state_dict' in state:
        gen.load_state_dict(state['model_state_dict'])
    else:
        gen.load_state_dict(state)

    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])

    img = Image.open(args.input).convert('L')
    x = transform(img).unsqueeze(0).to(device)
    gen.eval()
    with torch.no_grad():
        out = gen(x)
    # assume output in [0,1]
    arr = out.squeeze().cpu().numpy()
    arr = (arr * 255).astype('uint8')
    Image.fromarray(arr).save(args.output)
    print("Saved:", args.output)

if __name__ == '__main__':
    main()
