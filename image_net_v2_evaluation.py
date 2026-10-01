import os
os.environ.setdefault("LOCAL_RANK", "0")
import torch
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Swin repo imports
from config import get_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs",
    default=None,
    nargs='+',
    )
    return parser.parse_args()


def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu',weights_only=False)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # remove DDP prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

from PIL import Image

class ImageNetV2Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        for class_idx in range(1000):
            class_dir = os.path.join(root, str(class_idx))
            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                self.samples.append((path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -------------------------
    # Load config
    # -------------------------
    config = get_config(args)

    # -------------------------
    # Build model
    # -------------------------
    model = build_model(config)
    model.to(device)
    model.eval()

    # -------------------------
    # Load checkpoint
    # -------------------------
    load_checkpoint(model, args.ckpt)

    # -------------------------
    # Transforms (ImageNet standard)
    # -------------------------
    img_size = config.DATA.IMG_SIZE

    transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------------------------
    # Dataset
    # -------------------------
    dataset = ImageNetV2Dataset(args.data_path, transform=transform)
    #dataset.classes = [str(i) for i in range(1000)]
    #dataset.class_to_idx = {str(i): i for i in range(1000)}

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers)

    # -------------------------
    # Evaluation
    # -------------------------
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Top-1
            _, preds = outputs.topk(1, dim=1)
            top1_correct += (preds.squeeze() == labels).sum().item()

            # Top-5
            _, top5 = outputs.topk(5, dim=1)
            top5_correct += sum([labels[i] in top5[i] for i in range(len(labels))])

            total += labels.size(0)

    top1 = top1_correct / total * 100
    top5 = top5_correct / total * 100

    print(f"\nTop-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")


if __name__ == '__main__':
    main()
