import os
import torch
from models.diffusion import DiffusionModel
import argparse
from torch.utils.data import DataLoader
from dataloder import TrainingDataset, TestingDataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from evaluator import evaluation_model
import wandb
import torch.nn.functional as F
import math


def train(args, device):
    if args.epoch_start > 0:
        ckpt_path = f"{args.save_dir}/model_{args.epoch_start}.pth"
    else:
        ckpt_path = None

    model = DiffusionModel(device, ckpt_path, args)
    train_dataset = TrainingDataset(args.dataset_dir)
    test_dataset = TestingDataset(args.dataset_dir, filename=args.test_file)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']
    total_steps = math.ceil(len(train_dataset) / args.batch_size) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,  #
        num_warmup_steps=math.ceil(total_steps * 0.05),
        num_training_steps=total_steps,
        last_epoch=args.epoch_start)

    evaluator = evaluation_model()

    os.makedirs(args.save_dir, exist_ok=True)

    best_acc = 0

    for epoch in range(args.epoch_start, args.epoch_start + args.epochs):
        model.train()
        dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)
        total_loss = 0

        for image, label in tqdm(dataloader,
                                 mininterval=2,
                                 desc=f"Train Epoch {epoch}"):

            image = image.to(device)
            label = label.to(device)

            t = torch.randint(
                0,  #
                model.time_steps,
                (image.size(0), ),
                device=image.device)

            loss = model.p_losses(image, label, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss

        avg_loss = total_loss / len(dataloader)

        print(f"[ Train ] Epoch {epoch} loss: {avg_loss:.6f}")

        wandb.log({
            "loss": avg_loss,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if epoch % args.save_interval != 0:
            continue

        torch.save(model.state_dict(), f"{args.save_dir}/model_{epoch}.pth")

        model.eval()
        dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        total_acc = 0
        noise_mean_loss = 0
        noise_var_loss = 0

        for label in tqdm(dataloader,
                          mininterval=2,
                          desc=f"Eval Epoch {epoch}"):
            label = label.to(device)
            image, noise_mean, noise_var = model.sample((3, 64, 64), label)
            acc = evaluator.eval(image, label)

            total_acc += acc
            noise_mean_loss += F.mse_loss(noise_mean,
                                          torch.zeros_like(noise_mean))
            noise_var_loss += F.mse_loss(noise_var, torch.ones_like(noise_var))

        avg_acc = total_acc / len(dataloader)
        noise_mean_loss = noise_mean_loss / len(dataloader)
        noise_var_loss = noise_var_loss / len(dataloader)
        print(
            f"[ Eval ] Epoch {epoch} acc: {avg_acc:.6f}, noise_mean_loss: {noise_mean_loss:.6f}, noise_var_loss: {noise_var_loss:.6f}"
        )

        wandb.log({
            "acc": avg_acc,
            "epoch": epoch,
            "noise_mean": noise_mean_loss,
            "noise_var": noise_var_loss
        })

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), f"{args.save_dir}/model_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--time-steps", type=int, default=1000)

    parser.add_argument("--epoch-start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)

    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--test-file", type=str, default="test.json")

    parser.add_argument("--save-dir", type=str, default="saved_models")
    parser.add_argument("--save-interval", type=int, default=10)

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
