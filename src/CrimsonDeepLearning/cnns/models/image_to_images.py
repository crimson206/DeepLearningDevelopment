from CrimsonDeepLearning.cnns.blocks.sizepreservingblocks import SizePreservingBlock, SizePreservingBlocks
from CrimsonDeepLearning.cnns.blocks.downsampleblocks import ConvDownsampleBlock
from CrimsonDeepLearning.cnns.blocks.upsampleblocks import UpsampleBlocks
from CrimsonDeepLearning.cnns.blocks.embeddingblocks import CategoryEmbeddingBlock
import torch
import torch.nn as nn

class ConditionalImageToImage(nn.Module):
    def __init__(
            self,
            domain_sizes: list[int],
            input_channel: int,
            height: int, 
            width: int,
            down_hidden_sizes: list[int],
            res_hidden_sizes: list[int],
            up_hidden_sizes: list[int],
            output_channel: int,
        ):
        super(ConditionalImageToImage, self).__init__()
        """
        Shape:
            - image: (n_batch, input_channel, height, weight)
            - conditions: (n_batch, len(domain_sizes), height, weight)
            - output: (n_batch, output_channel, height, weight)
        """
        self.embeddings = CategoryEmbeddingBlock(domain_sizes=domain_sizes, height=height, width=width)
        self.initial = SizePreservingBlock(in_channel=input_channel+len(domain_sizes), out_channel=down_hidden_sizes[0], use_residual=False)
        self.downsample = ConvDownsampleBlock(down_hidden_sizes)
        self.res_blocks = SizePreservingBlocks(hidden_channels=res_hidden_sizes, use_residual=True)
        self.upsample = UpsampleBlocks(hidden_sizes=up_hidden_sizes)
        self.output = nn.Conv2d(up_hidden_sizes[-1], output_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, image: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:

        conditions = self.embeddings(conditions)

        image = torch.cat([image, conditions], dim=1)
        image = self.initial(image)
        image = self.downsample(image)
        image = self.res_blocks(image)
        image = self.upsample(image)
        image = self.output(image)

        return image