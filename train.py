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


def train(args, device):
    if args.epoch_start > 0:
        ckpt_path = f"{args.save_dir}/model_{args.epoch_start}.pth"
    else:
        ckpt_path = None

    model = DiffusionModel(device, ckpt_path, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5000,  # warmup 多少步
        num_training_steps=200000  # 總訓練步數
    )

    evaluator = evaluation_model()

    os.makedirs(args.save_dir, exist_ok=True)

    best_acc = 0

    for epoch in range(args.epoch_start, args.epoch_start + args.epochs):
        model.train()
        dataloader = DataLoader(TrainingDataset(args.dataset_dir),
                                batch_size=args.batch_size,
                                shuffle=True)
        total_loss = 0

        for image, label in tqdm(dataloader,
                                 mininterval=2,
                                 desc=f"Epoch {epoch}"):

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

        print(f"[ Train ] Epoch {epoch} loss: {avg_loss}")

        wandb.log({
            "loss": avg_loss,
            "epoch": epoch,
        })

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(),
                       f"{args.save_dir}/model_{epoch}.pth")

        if epoch % 10 != 0:
            continue

        model.eval()
        dataloader = DataLoader(  #
            TestingDataset(args.dataset_dir, filename=args.test_file),
            batch_size=args.batch_size)
        total_acc = 0

        for label in tqdm(dataloader, mininterval=2, desc=f"Epoch {epoch}"):
            label = label.to(device)
            image = model.sample((3, 64, 64), label)
            acc = evaluator.eval(image, label)
            total_acc += acc

        avg_acc = total_acc / len(dataloader)
        print(f"[ Eval ] Epoch {epoch} acc: {avg_acc}")

        wandb.log({
            "acc": avg_acc,
            "epoch": epoch,
        })

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), f"{args.save_dir}/model_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--time-steps", type=int, default=1000)

    parser.add_argument("--epoch-start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)

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
