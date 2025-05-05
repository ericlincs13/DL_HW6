import torch
import torch.nn as nn
from diffusers import UNet2DModel  # type: ignore


class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=24, class_emb_size=512):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Linear(num_classes, class_emb_size)

        channel = class_emb_size // 4

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=64,  # the target image resolution
            in_channels=3,  # Additional input channels for class cond.
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(
                channel,  #
                channel,
                channel * 2,
                channel * 2,
                channel * 4,
                channel * 4),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",  # a regular ResNet upsampling block
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            class_embed_type="identity")

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, class_labels):
        # class conditioning in right shape to add as additional input channels
        class_cond = self.class_emb(class_labels)  # Map to embedding dimension

        return self.model(x, t, class_cond).sample


if __name__ == "__main__":
    model = ClassConditionedUnet()
    print(model)
    print(
        model(torch.randn(1, 3, 64, 64), 10,
              torch.randint(0, 1, (1, 24), dtype=torch.float)).shape)
