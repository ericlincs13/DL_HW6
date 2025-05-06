import os
import torch
from torchvision.utils import save_image
import argparse
from models.diffusion import Diffusion
from torch.utils.data import DataLoader
from dataset import TestingDataset
from evaluator import evaluation_model
from tqdm import tqdm
from torchvision.utils import make_grid


def inference(args, device):
    model = Diffusion(device, args, mode="test")
    evaluator = evaluation_model()
    test_dataset_1 = TestingDataset(args.dataset_dir, filename='test.json')
    test_dataloader_1 = DataLoader(test_dataset_1,
                                   batch_size=1,
                                   num_workers=args.num_workers)
    test_dataset_2 = TestingDataset(args.dataset_dir, filename='new_test.json')
    test_dataloader_2 = DataLoader(test_dataset_2,
                                   batch_size=1,
                                   num_workers=args.num_workers)

    os.makedirs(args.output_dir, exist_ok=True)

    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'new_test'), exist_ok=True)

    total_acc = 0
    dir = os.path.join(args.output_dir, 'test')
    progress_bar = tqdm(test_dataloader_1, desc=f"Inference test.json")
    final_images = []

    for idx, label in enumerate(progress_bar):
        label = label.to(device)
        image, denoising = model.sample(label)
        acc = evaluator.eval(image, label)
        total_acc += acc

        image = (image + 1) / 2
        final_images.append(image)

        denoising = denoising.transpose(0, 1)
        denoising = (denoising + 1) / 2
        for i in range(denoising.shape[0]):
            img_idx = idx * denoising.shape[0] + i
            save_path = os.path.join(dir, f"{img_idx}.png")
            row_image = make_grid(denoising[i],
                                  nrow=denoising[i].shape[0],
                                  pad_value=0)
            save_image(row_image, save_path, normalize=False)

    final_images = torch.stack(final_images, dim=0)
    final_images = final_images.reshape(-1, *final_images.shape[2:])
    synthetic = make_grid(final_images,
                          nrow=len(final_images) // 4,
                          pad_value=0)
    save_path = os.path.join(dir, 'synthetic.png')
    save_image(synthetic, save_path, normalize=False)

    print(f"Avg acc of test.json: {total_acc / len(test_dataloader_1)}")

    total_acc = 0
    dir = os.path.join(args.output_dir, 'new_test')
    progress_bar = tqdm(test_dataloader_2, desc=f"Inference new_test.json")
    final_images = []

    for idx, label in enumerate(progress_bar):
        label = label.to(device)
        image, denoising = model.sample(label)
        acc = evaluator.eval(image, label)
        total_acc += acc

        image = (image + 1) / 2
        final_images.append(image)

        denoising = denoising.transpose(0, 1)
        denoising = (denoising + 1) / 2
        for i in range(denoising.shape[0]):
            img_idx = idx * denoising.shape[0] + i
            save_path = os.path.join(dir, f"{img_idx}.png")
            row_image = make_grid(denoising[i],
                                  nrow=denoising[i].shape[0],
                                  pad_value=0)
            save_image(row_image, save_path, normalize=False)

    final_images = torch.stack(final_images, dim=0)
    final_images = final_images.reshape(-1, *final_images.shape[2:])
    synthetic = make_grid(final_images,
                          nrow=len(final_images) // 4,
                          pad_value=0)
    save_path = os.path.join(dir, 'synthetic.png')
    save_image(synthetic, save_path, normalize=False)

    print(f"Avg acc of new_test.json: {total_acc / len(test_dataloader_2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=20)

    parser.add_argument("--dataset-dir", type=str, default="data")

    parser.add_argument("--ckpt-path",
                        type=str,
                        default="saved_models/model_best.pth")
    parser.add_argument("--output-dir", type=str, default="synthetic")

    parser.add_argument("--guidance-scale",
                        type=float,
                        default=1.5,
                        help="scale for classifier guidance")

    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    inference(args, device)
