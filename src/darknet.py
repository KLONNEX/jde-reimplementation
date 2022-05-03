"""DarkNet53 model."""
from torch import nn


def conv_bn_relu(
        in_channels,
        out_channels,
        kernel_size,
        stride,
):
    """
    Convolution (without bias), BatchNorm, ReLU activation block.

    Args:
        in_channels (int): Conv input feature map channels.
        out_channels (int): Conv output feature map channels.
        kernel_size (int): Conv kernel size.
        stride (int): Conv stride.

    Returns:
        block (nn.Sequential): Builded ConvBNReLU block.
    """
    padding = 'same'
    if stride > 1:
        padding = stride // 2

    block = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

    return block


class ResidualBlock(nn.Module):
    """
    Basic residual block with Conv, BN, ReLU.

    Args:
        in_channels (int): Number of channels in input feature map.
        out_channels (int): Number of channels in output feature map channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        hidden_channels = out_channels // 2  # Num of hidden filters

        self.conv1 = conv_bn_relu(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
        )

        self.conv2 = conv_bn_relu(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
        )

    def forward(self, inp):
        """
        Convolve input feature map, add skip connection.

        Args:
            inp (Tensor): Input feature map (bs, in_channels, H, W).

        Returns:
            out (Tensor): Output feature map (bs, out_channels, H, W).
        """
        skip_connect = inp

        hidden = self.conv1(inp)
        hidden = self.conv2(hidden)

        out = hidden + skip_connect

        return out


class DarkNet(nn.Module):
    """
    DarkNet53 model.

    Args:
        block (class): Sequential block.
        layer_nums (list): List with numbers of sequential blocks into layer.
        in_channels (list): List with numbers of input channels for the layers.
        out_channels (list): List with numbers of output channels for the layers.
        detect (bool): Use darknet as FPN.
    """
    def __init__(
            self,
            block,
            layer_nums,
            in_channels,
            out_channels,
            detect=True,
    ):
        super().__init__()

        self.detect = detect

        self.conv0 = conv_bn_relu(
            in_channels=3,
            out_channels=in_channels[0],
            kernel_size=3,
            stride=1,
        )

        self.conv1 = conv_bn_relu(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=3,
            stride=2,
        )

        self.layer1 = self._build_block(
            block=block,
            layer_num=layer_nums[0],
            in_channel=out_channels[0],
            out_channel=out_channels[0],
        )

        self.conv2 = conv_bn_relu(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=3,
            stride=2,
        )

        self.layer2 = self._build_block(
            block=block,
            layer_num=layer_nums[1],
            in_channel=out_channels[1],
            out_channel=out_channels[1],
        )

        self.conv3 = conv_bn_relu(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=3,
            stride=2,
        )

        self.layer3 = self._build_block(
            block=block,
            layer_num=layer_nums[2],
            in_channel=out_channels[2],
            out_channel=out_channels[2],
        )

        self.conv4 = conv_bn_relu(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=3,
            stride=2,
        )

        self.layer4 = self._build_block(
            block=block,
            layer_num=layer_nums[3],
            in_channel=out_channels[3],
            out_channel=out_channels[3],
        )

        self.conv5 = conv_bn_relu(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=3,
            stride=2,
        )

        self.layer5 = self._build_block(
            block=block,
            layer_num=layer_nums[4],
            in_channel=out_channels[4],
            out_channel=out_channels[4],
        )

    @staticmethod
    def _build_block(
            block,
            layer_num,
            in_channel,
            out_channel,
    ):
        """
        Build sequential block.

        Args:
            block (nn.Module): Residual block.
            layer_num (int): Number of blocks.
            in_channel (int): Number of channels in input feature map.
            out_channel (int): Number of channels in output feature map channels.

        Returns:
            darknet_block (nn.Sequential): Builded block.
        """
        darkblk = block(in_channel, out_channel)
        darknet_block = nn.Sequential(darkblk)

        for _ in range(1, layer_num):
            darkblk = block(out_channel, out_channel)
            darknet_block.append(darkblk)

        return darknet_block

    def forward(self, inp):
        """
        Feed forward input image.

        Args:
            inp (Tensor): Input image (bs, 3, H, W)

        Returns:
            c11 (Tensor): Output feature map.
        """
        c1 = self.conv0(inp)
        c2 = self.conv1(c1)
        c3 = self.layer1(c2)
        c4 = self.conv2(c3)
        c5 = self.layer2(c4)
        c6 = self.conv3(c5)
        c7 = self.layer3(c6)
        c8 = self.conv4(c7)
        c9 = self.layer4(c8)
        c10 = self.conv5(c9)
        c11 = self.layer5(c10)

        if self.detect:
            return c7, c9, c11

        return c11
