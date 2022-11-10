from torch import nn
from torch.ao.nn.quantized import FloatFunctional


class FirstOctave(nn.Module):
    """
        First octave convolution layer to create two output frequencies

        eq. (6)
    """

    def __init__(self, out_channels, activation=nn.LeakyReLU):
        super(FirstOctave, self).__init__()

        self.activation = activation()

        self.conv_h = nn.Conv2d(in_channels=3,
                                out_channels=out_channels // 2,
                                kernel_size=3,
                                stride=2,
                                padding=1)

        self.conv_l = nn.Conv2d(in_channels=out_channels // 2,
                                out_channels=out_channels // 2,
                                kernel_size=3,
                                stride=2,
                                padding=1)

    def forward(self, x):
        y_h = self.activation(self.conv_h(x))
        y_l = self.activation(self.conv_l(y_h))
        return y_h, y_l


class LastOctave(nn.Module):
    """
        Last octave convolution layer to fuse two output frequencies into an image

        eq. (7)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 alpha=0.5,
                 activation=nn.LeakyReLU):
        super(LastOctave, self).__init__()

        self.activation = activation()
        beta = 1. - alpha
        self.l_in_channels = int(alpha * in_channels)
        self.h_in_channels = int(beta * in_channels)
        self.add = FloatFunctional()

        self.conv_h = nn.ConvTranspose2d(in_channels=self.h_in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=2,
                                         padding=kernel_size // 2,
                                         output_padding=1)

        self.conv_l = nn.ConvTranspose2d(in_channels=self.l_in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=2,
                                         padding=kernel_size // 2,
                                         output_padding=1)

        self.upsample = nn.ConvTranspose2d(in_channels=3,
                                           out_channels=out_channels,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1)

    def forward(self, y):
        y_h, y_l = y
        x_h_h = self.conv_h(y_h)
        x_l_l = self.conv_l(y_l)
        x = self.add.add(x_h_h, self.upsample(x_l_l))
        return x


class GeneralizedOctaveConv(nn.Module):
    """
        eq. (4)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 alpha=0.5,
                 activation=nn.LeakyReLU,
                 norm=None):
        super(GeneralizedOctaveConv, self).__init__()
        self.add = FloatFunctional()

        self.activation = activation()
        beta = 1. - alpha
        self.l_in_channels = int(alpha * in_channels)
        self.h_in_channels = int(beta * in_channels)

        self.l_out_channels = int(alpha * out_channels)
        self.h_out_channels = int(beta * out_channels)
        self.upsample = nn.ConvTranspose2d(in_channels=self.l_out_channels,
                                           out_channels=self.l_out_channels,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1)
        self.downsample = nn.Conv2d(in_channels=self.h_out_channels,
                                    out_channels=self.h_out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.conv_l = nn.Conv2d(in_channels=self.l_in_channels,
                                out_channels=self.l_out_channels,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2,
                                stride=stride)
        self.conv_h = nn.Conv2d(in_channels=self.h_in_channels,
                                out_channels=self.h_out_channels,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2,
                                stride=stride)
        self.norm_h = norm(self.h_out_channels) if norm else nn.Identity()
        self.norm_l = norm(self.l_out_channels) if norm else nn.Identity()

    def forward(self, x):
        x_h, x_l = x
        y_l_l = self.activation(self.conv_l(x_l))
        y_h_h = self.activation(self.conv_h(x_h))
        y_h_l = self.activation(self.downsample(y_h_h))
        y_l_h = self.activation(self.upsample(y_l_l))
        y_l = self.add.add(y_h_l, y_l_l)
        y_h = self.add.add(y_l_h, y_h_h)
        return y_h, y_l


class GeneralizedOctaveTransposeConv(GeneralizedOctaveConv):
    """
        eq. (5)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 alpha=0.5,
                 activation=nn.LeakyReLU,
                 norm=None):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         alpha=alpha,
                         activation=activation,
                         norm=norm)
        self.conv_l = nn.ConvTranspose2d(in_channels=self.l_in_channels,
                                         out_channels=self.l_out_channels,
                                         kernel_size=kernel_size,
                                         padding=kernel_size // 2,
                                         output_padding=stride - 1,
                                         stride=stride)
        self.conv_h = nn.ConvTranspose2d(in_channels=self.h_in_channels,
                                         out_channels=self.h_out_channels,
                                         kernel_size=kernel_size,
                                         padding=kernel_size // 2,
                                         output_padding=stride - 1,
                                         stride=stride)
        self.norm_h = norm(self.h_out_channels) if norm else nn.Identity()
        self.norm_l = norm(self.l_out_channels) if norm else nn.Identity()

    def forward(self, x):
        x_h, x_l = x
        # I genuenly don't understand why they apply the activations to the input in the GoTConv.
        #  Because it looks cooler in the figures?
        y_l_l = self.conv_l(self.activation(x_l))
        y_h_h = self.conv_h(self.activation(x_h))
        y_h_l = self.downsample(self.activation(y_h_h))
        y_l_h = self.upsample(self.activation(y_l_l))
        y_l = self.add.add(y_h_l, y_l_l)
        y_h = self.add.add(y_l_h, y_h_h)
        return y_h, y_l
