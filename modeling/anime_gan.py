import torch
from torch.nn import init
import functools
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from modeling.conv_blocks import DownConv
from modeling.conv_blocks import UpConv
from modeling.conv_blocks import SeparableConv2D
from modeling.conv_blocks import InvertedResBlock
from modeling.conv_blocks import ConvBlock
from utils.common import initialize_weights


def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    init.xavier_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()
    return spectral_norm(module)

def leaky_relu(input):
    return F.leaky_relu(input, negative_slope=0.2)



class SelfAttention(nn.Module):
    def __init__(self, in_channel, gain=1):
        super().__init__()
        print(in_channel)
                                   
        self.query = spectral_init(nn.Conv1d(in_channel, in_channel, 1),
                                   gain=gain)
        self.key = spectral_init(nn.Conv1d(in_channel, in_channel , 1),
                                 gain=gain)
        self.value = spectral_init(nn.Conv1d(in_channel, in_channel, 1),
                                   gain=gain)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

        return out



class Generator(nn.Module):
    def __init__(self, dataset=''):
        super(Generator, self).__init__()
        self.name = f'generator_{dataset}'
        bias = False

        self.encode_blocks = nn.Sequential(
            ConvBlock(3, 64, bias=bias),
            ConvBlock(64, 128, bias=bias),
            DownConv(128, bias=bias),
            ConvBlock(128, 128, bias=bias),
            SeparableConv2D(128, 256, bias=bias),
            DownConv(256, bias=bias),
            ConvBlock(256, 256, bias=bias),
        )

        self.res_blocks = nn.Sequential(
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
        )

        self.decode_blocks = nn.Sequential(
            ConvBlock(256, 128, bias=bias),
            UpConv(128, bias=bias),
            SeparableConv2D(128, 128, bias=bias),
            ConvBlock(128, 128, bias=bias),
            UpConv(128, bias=bias),
            ConvBlock(128, 64, bias=bias),
            ConvBlock(64, 64, bias=bias),
            ConvBlock(64, 3, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Tanh(),
        )
        # add attention
        self.attn1 = SelfAttention(128)
        self.attn2 = SelfAttention(64)

        initialize_weights(self)

    def forward(self, x):
        out = self.encode_blocks(x)
        out = self.res_blocks(out)
        img = self.decode_blocks(out)
        

        return img


class Discriminator(nn.Module):
    def __init__(self,  args):
        super(Discriminator, self).__init__()
        self.name = f'discriminator_{args.dataset}'
        self.bias = False
        channels = 32

        layers = [
            nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, True)
        ]

        for i in range(args.d_layers):
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=self.bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1, bias=self.bias),
                nn.InstanceNorm2d(channels * 4),
                nn.LeakyReLU(0.2, True),
            ]
            channels *= 4

        layers += [
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=self.bias),
        ]

        if args.use_sn:
            for i in range(len(layers)):
                if isinstance(layers[i], nn.Conv2d):
                    layers[i] = spectral_norm(layers[i])

        self.discriminate = nn.Sequential(*layers)
        # add attention
        self.attn1 = SelfAttention(channels)

        initialize_weights(self)

    def forward(self, img):
        return self.discriminate(img)
