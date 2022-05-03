from torch import nn


def _conv_bn_leaky(
        in_channels,
        out_channels,
        kernel_size,
):
    """
    Set a conv2d, BN and leaky relu layer.

    Args:
        in_channels (int): Conv input feature map channels.
        out_channels (int): Conv output feature map channels.
        kernel_size (int): Conv kernel size.

    Returns:
        block (nn.Sequential): Builded block.
    """
    block = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same',
            bias=False,
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(negative_slope=0.1),
    )

    return block


class YoloBlock(nn.Module):
    """
    YoloBlock for YOLOv3.

    Args:
        in_channels (int): Input channel.
        out_chls (int): Middle channel.
        out_channels (int): Output channel.
        emb_dim (int): Embedding size.

    Returns:
        c5 (Tensor): Feature map to feed at next layers.
        out (Tensor): Output feature map.
        emb (Tensor): Output embeddings.
    """

    def __init__(
            self,
            in_channels,
            out_chls,
            out_channels,
            emb_dim=512,
    ):
        super().__init__()
        out_chls_2 = out_chls * 2

        self.conv0 = _conv_bn_leaky(in_channels, out_chls, kernel_size=1)
        self.conv1 = _conv_bn_leaky(out_chls, out_chls_2, kernel_size=3)

        self.conv2 = _conv_bn_leaky(out_chls_2, out_chls, kernel_size=1)
        self.conv3 = _conv_bn_leaky(out_chls, out_chls_2, kernel_size=3)

        self.conv4 = _conv_bn_leaky(out_chls_2, out_chls, kernel_size=1)
        self.conv5 = _conv_bn_leaky(out_chls, out_chls_2, kernel_size=3)

        self.conv6 = nn.Conv2d(out_chls_2, out_channels, kernel_size=1, padding='same', bias=False)

        self.emb_conv = nn.Conv2d(out_chls, emb_dim, kernel_size=3, padding='same', bias=False)

    def forward(self, x):
        """
        Feed forward feature map to YOLOv3 block
        to get detections and embeddings.
        """
        c1 = self.conv0(x)
        c2 = self.conv1(c1)

        c3 = self.conv2(c2)
        c4 = self.conv3(c3)

        c5 = self.conv4(c4)
        c6 = self.conv5(c5)

        emb = self.emb_conv(c5)

        out = self.conv6(c6)

        return c5, out, emb
