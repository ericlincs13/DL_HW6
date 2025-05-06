import os
import torch
from models.diffusion import Diffusion
import argparse
import wandb


def train(args, device):
    diffusion = Diffusion(device, args, mode="train")

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0
    for epoch in range(args.epoch_start, args.epoch_start + args.epochs):
        avg_loss = diffusion.train(epoch)

        print(f"[ Train ] Epoch {epoch} loss: {avg_loss}")

        if epoch % args.save_interval != 0:
            continue

        diffusion.save_ckpt(f"model_{epoch}")

        avg_acc = diffusion.eval(epoch)

        print(f"[ Eval ] Epoch {epoch} acc: {avg_acc}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            diffusion.save_ckpt("model_best")

    diffusion.save_ckpt("model_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--time-steps", type=int, default=1000)

    parser.add_argument("--epoch-start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=20)

    parser.add_argument("--dataset-dir", type=str, default="data")
    parser.add_argument("--test-file", type=str, default="test.json")

    parser.add_argument("--save-dir", type=str, default="saved_models")
    parser.add_argument("--save-interval", type=int, default=10)

    parser.add_argument("--use-scheduler",
                        type=str,
                        default='cosine',
                        choices=['warmup', 'cosine', 'none'])
    parser.add_argument("--uncond-prob",
                        type=float,
                        default=0.1,
                        help="probability of dropping class conditioning")
    parser.add_argument("--guidance-scale",
                        type=float,
                        default=1.5,
                        help="scale for classifier guidance")

    parser.add_argument("--wandb-run-name",
                        type=str,
                        default="diffusion-train")
    parser.add_argument("--wandb-dir", type=str)

    args = parser.parse_args()

    wandb.init(project="DLP-Lab6-Diffusion",
               name=args.wandb_run_name,
               dir=args.wandb_dir,
               save_code=True)

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train(args, device)
