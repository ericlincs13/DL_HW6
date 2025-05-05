import os
import torch
from torchvision.utils import save_image
import argparse
from models.diffusion import Diffusion
from torch.utils.data import DataLoader
from dataloder import TestingDataset
from tqdm import tqdm
from evaluator import evaluation_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path",
                        type=str,
                        default="saved_models/model_best.pth")
    parser.add_argument("--save-dir", type=str, default="synthetic")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=1000)
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--test-file", type=str, default="test.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Diffusion(device, args)
    evaluator = evaluation_model()
    dataset = TestingDataset(args.dataset_dir, filename=args.test_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    os.makedirs(args.save_dir, exist_ok=True)

    total_acc = 0
    idx = 0
    for label in tqdm(dataloader, mininterval=2, desc=f"Eval"):
        label = label.to(device)
        image = model.sample(label)
        acc = evaluator.eval(image, label)
        total_acc += acc

        for i in range(args.batch_size):
            img_idx = idx * args.batch_size + i
            save_path = os.path.join(args.save_dir, f"{img_idx}.png")
            save_image(image[i], save_path, normalize=False)

        idx += 1

    print(f"Avg acc: {total_acc / len(dataloader)}")
