import torch
import torch.nn as nn
from diffusers import DDPMScheduler  # type: ignore
from models.Unet import ClassConditionedUnet
from dataloder import TrainingDataset, TestingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluator import evaluation_model
import wandb
from transformers import get_cosine_schedule_with_warmup
import math


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Diffusion():
    def __init__(self, device, args):
        self.device = device
        self.batch_size = args.batch_size

        self.uncond_prob = args.uncond_prob
        self.guidance_scale = args.guidance_scale
        self.time_steps = args.time_steps
        self.save_dir = args.save_dir

        self.model = ClassConditionedUnet().to(self.device)
        if args.epoch_start > 0:
            self.load_ckpt(f"{args.save_dir}/model_{args.epoch_start}.pth")
        else:
            self.model.apply(init_weights)

        self.evaluator = evaluation_model()

        train_dataset = TrainingDataset(args.dataset_dir)
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers)

        test_dataset = TestingDataset(args.dataset_dir,
                                      filename=args.test_file)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=32,
                                          num_workers=args.num_workers)

        self.loss_fn = nn.MSELoss()

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,  #
            beta_schedule="squaredcos_cap_v2")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        for pg in self.optimizer.param_groups:
            pg['initial_lr'] = pg['lr']

        total_steps = math.ceil(
            len(train_dataset) / args.batch_size) * args.epochs
        if args.use_scheduler:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,  #
                num_warmup_steps=math.ceil(total_steps * 0.05),
                num_training_steps=total_steps,
                last_epoch=args.epoch_start)
        else:
            self.scheduler = None

    def train(self, epoch):
        self.model.train()
        loss_list = []
        progress_bar = tqdm(self.train_dataloader, desc=f"Train Epoch {epoch}")
        for image, label in progress_bar:
            image = image.to(self.device)
            label = label.to(self.device)

            noise = torch.randn_like(image)
            t = torch.randint(0, self.time_steps, (image.shape[0], ))
            t = t.long().to(self.device)
            image_noisy = self.noise_scheduler.add_noise(  #
                image, noise, t)  # type: ignore

            # classifier-free guidance training: randomly drop class conditioning
            if self.uncond_prob > 0:
                mask = torch.rand(image.shape[0],
                                  device=self.device) < self.uncond_prob
                label_input = label.clone()
                label_input[mask] = 0.0
            else:
                label_input = label
            predict = self.model(image_noisy, t, label_input)
            loss = self.loss_fn(predict, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            loss_list.append(loss.item())

        avg_loss = sum(loss_list) / len(loss_list)

        wandb.log({
            "loss": avg_loss,
            "epoch": epoch,
            "lr": self.optimizer.param_groups[0]["lr"]
        })

        return avg_loss

    @torch.no_grad()
    def eval(self, epoch):
        self.model.eval()
        acc_list = []
        progress_bar = tqdm(self.test_dataloader, desc=f"Eval Epoch {epoch}")
        for label in progress_bar:
            label = label.to(self.device)
            image = self.sample(label)
            acc = self.evaluator.eval(image, label)
            acc_list.append(acc)

        avg_acc = sum(acc_list) / len(acc_list)

        wandb.log({
            "acc": avg_acc,
            "epoch": epoch,
        })

        return avg_acc

    @torch.no_grad()
    def sample(self, label):
        self.model.eval()
        batch_size = label.shape[0]
        image = torch.randn((batch_size, 3, 64, 64)).to(self.device)

        for t in self.noise_scheduler.timesteps:
            # classifier-free guidance sampling: combine unconditional and conditional predictions
            uncond_label = torch.zeros_like(label)
            noise_uncond = self.model(image, t, uncond_label)
            noise_cond = self.model(image, t, label)
            noise = noise_uncond + self.guidance_scale * (noise_cond -
                                                          noise_uncond)
            image = self.noise_scheduler.step(
                noise, t, image).prev_sample  # type: ignore

        return image

    def save_ckpt(self, filename):
        torch.save(
            {
                "model":
                self.model.state_dict(),
                "optimizer":
                self.optimizer.state_dict(),
                "scheduler":
                self.scheduler.state_dict()
                if self.scheduler is not None else None,
            }, f"{self.save_dir}/{filename}.pth")

    def load_ckpt(self, filename):
        ckpt = torch.load(f"{self.save_dir}/{filename}.pth")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
