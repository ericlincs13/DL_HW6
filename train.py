import os
import torch
from models.diffusion import DiffusionModel
import argparse
from torch.utils.data import DataLoader
from dataloder import TrainingDataset, TestingDataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from evaluator import evaluation_model


def train(args, device):
    model = DiffusionModel(device, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5000,  # warmup 多少步
        num_training_steps=200000  # 總訓練步數
    )

    evaluator = evaluation_model()

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epoch_start, args.epoch_start + args.epochs):
        model.train()
        dataloader = DataLoader(TrainingDataset(args.train_dir),
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

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(),
                       f"{args.save_dir}/model_{epoch}.pth")

        if epoch % 10 != 0:
            continue

        model.eval()
        dataloader = DataLoader(TestingDataset(args.test_file),
                                batch_size=args.batch_size)
        total_acc = 0

        for image, label in tqdm(dataloader,
                                 mininterval=2,
                                 desc=f"Epoch {epoch}"):

            image = image.to(device)
            label = label.to(device)
            acc = evaluator.eval(image, label)
            total_acc += acc

        avg_acc = total_acc / len(dataloader)
        print(f"[ Eval ] Epoch {epoch} acc: {avg_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--time-steps", type=int, default=1000)

    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--epoch-start", type=int, default=0)

    parser.add_argument("--train-dir", type=str, default="dataset")
    parser.add_argument("--test-file", type=str, default="test.json")

    parser.add_argument("--save-dir", type=str, default="saved_models")
    parser.add_argument("--save-interval", type=int, default=10)

    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train(args, device)
