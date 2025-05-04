import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1),  #
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.emb_mlp = nn.Linear(emb_dim, out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        out = self.conv1(x)
        out += self.emb_mlp(emb).unsqueeze(-1).unsqueeze(-1)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out / 1.414  # normalize


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 最終輸出的 embedding 維度

    def forward(self, timesteps):
        """
        timesteps: (B,) 的整數張量，代表時間步
        回傳: (B, dim) 的 sinusoidal embedding 向量
        """
        device = timesteps.device
        half_dim = self.dim // 2

        # 建立 dim/2 長度的 log-scale 頻率向量
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # shape: (half_dim,)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # 對每個 t 計算 [sin(t * freq), cos(t * freq)]
        emb = timesteps[:, None] * emb[None, :]  # shape: (B, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # shape: (B, dim)

        return emb


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, base_channels, emb_dim, layers=4):
        super(UNetEncoder, self).__init__()

        self.init = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),  #
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = [base_channels * (2**i) for i in range(layers)]

        for i in range(layers):
            if i == 0:
                self.blocks.append(
                    ResidualBlock(channels[i], channels[i], emb_dim))
                self.blocks.append(
                    ResidualBlock(channels[i], channels[i], emb_dim))

            else:
                self.blocks.append(
                    ResidualBlock(channels[i] // 2, channels[i], emb_dim))
                self.blocks.append(
                    ResidualBlock(channels[i], channels[i], emb_dim))

            self.downs.append(
                nn.Conv2d(channels[i],
                          channels[i],
                          kernel_size=3,
                          stride=2,
                          padding=1))

    def forward(self, x, emb):
        skips = []
        x = self.init(x)
        for i in range(len(self.downs)):
            x = self.blocks[i * 2](x, emb)
            x = self.blocks[i * 2 + 1](x, emb)
            x = self.downs[i](x)
            skips.append(x)
        return x, skips


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, base_channels, emb_dim, layers=4):
        super(UNetDecoder, self).__init__()

        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels * (2**i) for i in reversed(range(layers))]

        for i in range(layers):
            self.ups.append(
                nn.ConvTranspose2d(channels[i] * 2,
                                   channels[i],
                                   kernel_size=2,
                                   stride=2))
            self.blocks.append(  #
                ResidualBlock(channels[i], channels[i] // 2, emb_dim))
            self.blocks.append(  #
                ResidualBlock(channels[i] // 2, channels[i] // 2, emb_dim))

        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x, skip, emb):
        for i in range(len(self.ups)):
            x = torch.cat([skip.pop(), x], dim=1)
            x = self.ups[i](x)
            x = self.blocks[i * 2](x, emb)
            x = self.blocks[i * 2 + 1](x, emb)
        return self.final(x)


class UNet(nn.Module):
    def __init__(
            self,  #
            in_channels,
            out_channels,
            base_channels,
            time_emb_dim,
            class_num):

        super(UNet, self).__init__()

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.class_emb = nn.Linear(class_num, time_emb_dim)

        self.encoder = UNetEncoder(in_channels, base_channels, time_emb_dim)
        self.middle = ResidualBlock(base_channels * 8, base_channels * 8,
                                    time_emb_dim)
        self.decoder = UNetDecoder(out_channels, base_channels, time_emb_dim)

    def forward(self, x, t, class_label):
        emb = self.time_emb(t) + self.class_emb(class_label)
        x, skips = self.encoder(x, emb)
        x = self.middle(x, emb)
        return self.decoder(x, skips, emb)


class DiffusionModel(nn.Module):
    def __init__(self, device, ckpt_path, args):
        super(DiffusionModel, self).__init__()
        self.device = device

        self.model = UNet(in_channels=3,
                          out_channels=3,
                          base_channels=64,
                          time_emb_dim=128,
                          class_num=24).to(device)

        self.sample_transform = transforms.Normalize(  #
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.time_steps = args.time_steps
        self.betas = torch.linspace(0.0001, 0.008, self.time_steps).to(device)
        self.alphas = 1.0 - self.betas

        a_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sq_a_cumprod = torch.sqrt(a_cumprod).to(device)
        self.sq_co_a_cumprod = torch.sqrt(1.0 - a_cumprod).to(device)

        if ckpt_path:
            self.load_state_dict(torch.load(ckpt_path))
        else:
            self.apply(init_weights)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0).to(self.device)

        sq_a = self.sq_a_cumprod[t].view(-1, 1, 1, 1)
        sq_co_a = self.sq_co_a_cumprod[t].view(-1, 1, 1, 1)

        return sq_a * x0 + sq_co_a * noise

    def p_losses(self, x0, class_label, t):
        noise = torch.randn_like(x0).to(self.device)
        xt = self.q_sample(x0, t, noise)
        predicted_noise = self.model(xt, t, class_label)
        mse = F.mse_loss(noise, predicted_noise, reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1)
        a_cumprod_t = (self.sq_a_cumprod[t]**2)
        weight = (self.betas[t]**2) / (self.alphas[t] * (1.0 - a_cumprod_t))
        loss = (weight * mse).mean()
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, class_label):
        b = self.betas[t].view(-1, 1, 1, 1)
        sq_a = torch.sqrt(self.alphas[t]).view(-1, 1, 1, 1)
        sq_co_a = self.sq_co_a_cumprod[t].view(-1, 1, 1, 1)

        predicted = self.model(x, t, class_label)
        mean = (1 / (sq_a + 1e-8)) * (x - b * predicted / (sq_co_a + 1e-8))

        if t[0] == 0:
            return mean, predicted
        else:
            noise = torch.randn_like(x)
            result = mean + torch.sqrt(b) * noise
            return result, predicted

    @torch.no_grad()
    def sample(self, shape, class_label):
        b = class_label.shape[0]
        x = torch.randn((b, *shape)).to(self.device)
        total_noise_mean = 0
        total_noise_var = 0
        for i in reversed(range(self.time_steps)):
            t = torch.full((b, ), i, device=x.device, dtype=torch.long)
            x, noise = self.p_sample(x, t, class_label)
            total_noise_mean = torch.mean(noise, dim=0)
            total_noise_var += torch.var(total_noise_mean)

        x = self.sample_transform(x)
        return x, total_noise_mean / self.time_steps, total_noise_var / self.time_steps
