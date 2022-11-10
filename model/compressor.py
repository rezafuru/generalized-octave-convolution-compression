import torch
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from torch import nn
from compressai.models import FactorizedPrior, JointAutoregressiveHierarchicalPriors, MeanScaleHyperprior, \
    get_scale_table

from model.layers import FirstOctave, GeneralizedOctaveConv, GeneralizedOctaveTransposeConv, LastOctave


class HighLowEntropyBottleneck(nn.Module):
    """
        They consider separate rate terms for x_l and x_h eq (16)-(18)

        I don't see why and in my own quick experiments I get better results if I re-combine low and high before compression
    """

    def __init__(self, entropy_bottleneck_channels, alpha=0.5):
        super(HighLowEntropyBottleneck, self).__init__()
        beta = 1. - alpha
        eb_l_ch = int(alpha * entropy_bottleneck_channels)
        eb_h_ch = int(beta * entropy_bottleneck_channels)
        self.eb_l = EntropyBottleneck(channels=eb_l_ch)
        self.eb_h = EntropyBottleneck(channels=eb_h_ch)

    def forward(self, x):
        x_h, x_l = x
        x_q_h, x_h_likelihoods = self.eb_h(x_h)
        x_q_l, x_l_likelihoods = self.eb_l(x_l)
        return x_q_h, x_q_l, x_h_likelihoods, x_l_likelihoods

    def compress(self, x):
        x_h, x_l = x
        x_h_bits = self.eb_h.compress(x_h)
        x_l_bits = self.eb_l.compress(x_l)
        return x_h_bits, x_l_bits

    def decompress(self, strings, sizes):
        x_h_string = self.eb_h.decompress(strings[0], sizes[0])
        x_l_string = self.eb_h.decompress(strings[1], sizes[1])
        return x_h_string, x_l_string

    def update(self, force=False):
        return self.eb_l.update(force=force) & self.eb_h.update(force=force)


class GeneralizedOctaveConvFP(FactorizedPrior):
    """
        Compression Model with Factorized Prior https://arxiv.org/pdf/1611.01704.pdf

        Entropy Model implementation from CompressAI
    """

    def __init__(self, n=192, m=192):
        super(GeneralizedOctaveConvFP, self).__init__(N=n, M=m)
        self.entropy_bottleneck = HighLowEntropyBottleneck(entropy_bottleneck_channels=m)
        self.g_a = nn.Sequential(
            FirstOctave(out_channels=n),
            GeneralizedOctaveConv(in_channels=n,
                                  out_channels=n,
                                  kernel_size=5,
                                  stride=2),
            GeneralizedOctaveConv(in_channels=n,
                                  out_channels=n,
                                  kernel_size=5,
                                  stride=2),
            GeneralizedOctaveConv(in_channels=n,
                                  out_channels=m,
                                  kernel_size=5,
                                  stride=2)
        )

        self.g_s = nn.Sequential(
            GeneralizedOctaveTransposeConv(in_channels=m,
                                           out_channels=n,
                                           kernel_size=5,
                                           stride=2),
            GeneralizedOctaveTransposeConv(in_channels=n,
                                           out_channels=n,
                                           kernel_size=5,
                                           stride=2),
            GeneralizedOctaveTransposeConv(in_channels=n,
                                           out_channels=n,
                                           kernel_size=5,
                                           stride=2),
            LastOctave(in_channels=n,
                       kernel_size=5,
                       out_channels=3)
        )

    def forward(self, x, return_likelihoods=False):
        y_h_l = self.g_a(x)
        y_q_h, y_q_l, y_h_likelihoods, y_l_likelihoods = self.entropy_bottleneck(y_h_l)
        x_hat = self.g_s((y_q_h, y_q_l))
        if return_likelihoods:
            return x_hat, {"y_h_likelihoods": y_h_likelihoods, "y_l_likelihoods": y_l_likelihoods}
        else:
            return x_hat

    def compress(self, x):
        y_h, y_l = self.g_a(x)
        y_h_bits, y_l_bits = self.entropy_bottleneck.compress((y_h, y_l))
        return {"strings": [y_h_bits, y_l_bits], "shapes": [y_h.size()[-2:], y_l.size()[-2:]]}

    def decompress(self, strings, shapes):
        assert isinstance(strings, list) and len(strings) == 2
        assert isinstance(shapes, list) and len(shapes) == 2
        y_h_l_hat = self.entropy_bottleneck.decompress(strings, shapes)
        x_hat = self.g_s(y_h_l_hat).clamp(0, 1)
        return x_hat

    def update(self, force=False):
        return self.entropy_bottleneck.update(force=force)


class GeneralizedOctaveConvMSHP(MeanScaleHyperprior):
    """
        Compression Model with Hyper Prior https://arxiv.org/abs/1802.01436, https://arxiv.org/abs/1809.02736

         Entropy Model implementation from CompressAI
    """

    def __init__(self, n=192, m=192):
        super(GeneralizedOctaveConvMSHP, self).__init__(N=n, M=m)
        delattr(self, 'gaussian_conditional')
        self.entropy_bottleneck = HighLowEntropyBottleneck(entropy_bottleneck_channels=n)
        self.g_a = nn.Sequential(
            FirstOctave(out_channels=n),
            GeneralizedOctaveConv(in_channels=n,
                                  out_channels=n,
                                  kernel_size=5,
                                  stride=2),
            GeneralizedOctaveConv(in_channels=n,
                                  out_channels=n,
                                  kernel_size=5,
                                  stride=2),
            GeneralizedOctaveConv(in_channels=n,
                                  out_channels=m,
                                  kernel_size=5,
                                  stride=2)
        )

        self.g_s = nn.Sequential(
            GeneralizedOctaveTransposeConv(in_channels=m,
                                           out_channels=n,
                                           kernel_size=5,
                                           stride=2),
            GeneralizedOctaveTransposeConv(in_channels=n,
                                           out_channels=n,
                                           kernel_size=5,
                                           stride=2),
            GeneralizedOctaveTransposeConv(in_channels=n,
                                           out_channels=n,
                                           kernel_size=5,
                                           stride=2),
            LastOctave(in_channels=n,
                       kernel_size=5,
                       out_channels=3)
        )

        self.h_a = nn.Sequential(
            GeneralizedOctaveConv(in_channels=m,
                                  out_channels=n,
                                  kernel_size=5,
                                  stride=1),
            GeneralizedOctaveConv(in_channels=n,
                                  out_channels=n,
                                  kernel_size=5,
                                  stride=2),
            GeneralizedOctaveConv(in_channels=n,
                                  out_channels=n,
                                  kernel_size=5,
                                  stride=2)
        )
        self.h_s = nn.Sequential(
            GeneralizedOctaveTransposeConv(in_channels=n,
                                           out_channels=m,
                                           kernel_size=5,
                                           stride=2),
            GeneralizedOctaveTransposeConv(in_channels=m,
                                           out_channels=m * 3 // 2,
                                           kernel_size=5,
                                           stride=2),

            GeneralizedOctaveTransposeConv(in_channels=m * 3 // 2,
                                           out_channels=m * 2,
                                           kernel_size=5,
                                           stride=1),
        )

        self.gaussian_conditional_h = GaussianConditional(None)
        self.gaussian_conditional_l = GaussianConditional(None)

    def forward(self, x, return_likelihoods=False):
        y_h, y_l = self.g_a(x)
        z_h_l = self.h_a((y_h, y_l))
        z_q_h, z_q_l, z_h_likelihoods, z_l_likelihoods = self.entropy_bottleneck(z_h_l)
        gaussian_params_h, gaussian_params_l = self.h_s((z_q_h, z_q_l))
        scales_h, means_h = gaussian_params_h.chunk(2, 1)
        scales_l, means_l = gaussian_params_l.chunk(2, 1)
        y_h_hat, y_h_likelihoods = self.gaussian_conditional_l(y_h, scales_h, means=means_h)
        y_l_hat, y_l_likelihoods = self.gaussian_conditional_l(y_l, scales_l, means=means_l)
        x_hat = self.g_s((y_h_hat, y_l_hat))
        if return_likelihoods:
            return x_hat, {"y_h_likelihoods": y_h_likelihoods,
                           "y_l_likelihoods": y_l_likelihoods,
                           "z_h_likelihoods": z_h_likelihoods,
                           "z_l_likelihoods": z_l_likelihoods}
        else:
            return x_hat

    def compress(self, x):
        y_h, y_l = self.g_a(x)
        z_h, z_l = self.h_a((y_h, y_l))
        z_h_bits, z_l_bits = self.entropy_bottleneck.compress((z_h, z_l))
        z_h_hat, z_l_hat = self.entropy_bottleneck.decompress((z_h_bits, z_l_bits), (z_h.size()[-2:], z_l.size()[-2:]))
        gaussian_params_h, gaussian_params_l = self.h_s((z_h_hat, z_l_hat))
        scales_h, means_h = gaussian_params_h.chunk(2, 1)
        scales_l, means_l = gaussian_params_l.chunk(2, 1)
        indexes_h = self.gaussian_conditional_h.build_indexes(scales_h)
        indexes_l = self.gaussian_conditional_l.build_indexes(scales_l)
        y_h_bits = self.gaussian_conditional_h.compress(y_h, indexes_h, means=means_h)
        y_l_bits = self.gaussian_conditional_l.compress(y_l, indexes_l, means=means_l)
        return {"strings": [[z_h_bits, z_l_bits], [y_h_bits, y_l_bits]],
                "shapes": [z_h.size()[-2:], z_l.size()[-2:]]
                }

    def decompress(self, strings, shapes):
        assert isinstance(strings, list) and len(strings) == 2
        assert isinstance(shapes, list) and len(shapes) == 2
        z_hat_h, z_hat_l = self.entropy_bottleneck.decompress(strings[0], shapes)
        gaussian_params_h, gaussian_params_l = self.h_s((z_hat_h, z_hat_l))
        scales_h, means_h = gaussian_params_h.chunk(2, 1)
        scales_l, means_l = gaussian_params_l.chunk(2, 1)
        indexes_h = self.gaussian_conditional_h.build_indexes(scales_h)
        indexes_l = self.gaussian_conditional_l.build_indexes(scales_l)
        strings_y_h, strings_y_l = strings[0]
        y_h = self.gaussian_conditional_h.decompress(strings_y_h, indexes_h, means=means_h)
        y_l = self.gaussian_conditional_l.decompress(strings_y_l, indexes_l, means=means_l)
        x_hat = self.g_s((y_h, y_l)).clamp(0, 1)
        return x_hat

    def update(self, scale_table_h= None, scale_table_l = None, force=False):
        scale_table_h = scale_table_h or get_scale_table()
        scale_table_l = scale_table_l or get_scale_table()
        updated = self.gaussian_conditional_h.update_scale_table(scale_table_h, force=force)
        updated |= self.gaussian_conditional_l.update_scale_table(scale_table_l, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated


class GeneralizedOctaveConvJAHP(JointAutoregressiveHierarchicalPriors):
    """
        Compression Model with context model and hyper prior https://arxiv.org/abs/1809.02736

        Entropy model implementation from CompressAI

        todo: CBA, since the authors aren't really proposing any novel context model, i.e. just follow the same
          pattern as the FP and MSHP implementation, and replace the sub-networks with the octave versions
          in the JointAutoregressiveHierarchicalPriors implementation of compressAI
    """


if __name__ == '__main__':
    x_shape = ()
    x = torch.randn(1, 3, 256, 256)
    m1 = GeneralizedOctaveConvFP()
    m2 = GeneralizedOctaveConvMSHP()
    x_t_1 = m1(x)
    x_t_2 = m2(x)
    m1.update()
    m2.update()
    x_c_1 = m1.compress(x)
    x_c_2 = m2.compress(x)
    x_dc_1 = m1.decompress(x_c_1["strings"], x_c_1["shapes"])
    x_dc_2 = m2.decompress(x_c_2["strings"], x_c_2["shapes"])
    assert x.shape == x_t_1.shape == x_dc_1.shape == x_t_2.shape == x_dc_2.shape
