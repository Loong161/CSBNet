import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)

##########################################################################
class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.fusion = conv(channel, channel, kernel_size=3, stride=1)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.fusion(x)
        return out


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (DAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(out_channels, out_channels, kernel_size, bias=bias))

        self.CA = CALayer(out_channels, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img

##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class DownFocus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownFocus, self).__init__()
        self.down = nn.Conv2d(in_channels * 4, out_channels, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.down(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
## U-Net
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias):
        super(Encoder, self).__init__()

        self.encoder = [CAB(n_feat, n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder = nn.Sequential(*self.encoder)

        self.down = DownSample(n_feat)
        self.conv = conv(n_feat, n_feat * 2, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x_down = self.conv(self.down(x))
        return x, x_down


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias):
        super(Decoder, self).__init__()

        self.decoder = [CAB(n_feat * 2, n_feat * 2, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder = nn.Sequential(*self.decoder)

        self.skip_attn = CAB(n_feat, n_feat, kernel_size, reduction, bias=bias, act=act)
        self.up = SkipUpSample(n_feat)

        self.conv = conv(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, x, y):
        x = self.decoder(x)
        x_up = self.conv(x)
        x_up = self.up(x_up, self.skip_attn(y))
        return x, x_up


##########################################################################
#Spacial info recovery
class SIRNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(SIRNet, self).__init__()
        self.orb1 = CAB(n_feat, n_feat, kernel_size, reduction, bias=bias, act=act)
        self.orb2 = CAB(n_feat, n_feat, kernel_size, reduction, bias=bias, act=act)
        self.orb3 = CAB(n_feat, n_feat, kernel_size, reduction, bias=bias, act=act)

        self.up_enc1 = UpSample(n_feat * 2, n_feat * 2)
        self.up_dec1 = UpSample(n_feat * 2, n_feat * 2)

        self.up_enc2 = nn.Sequential(UpSample(n_feat * 4, n_feat * 4),
                                     UpSample(n_feat * 4, n_feat * 4))
        self.up_dec2 = nn.Sequential(UpSample(n_feat * 4, n_feat * 4),
                                     UpSample(n_feat * 4, n_feat * 4))

        self.conv_1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_2 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)
        self.conv_3 = nn.Conv2d(n_feat * 4, n_feat, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_1(encoder_outs[0]) + self.conv_1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_2(self.up_enc1(encoder_outs[1])) + self.conv_2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_3(self.up_enc2(encoder_outs[2])) + self.conv_3(self.up_dec2(decoder_outs[2]))

        return x

##########################################################################
# Contextual info recovery
class CIRNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CIRNet, self).__init__()
        self.feat_extract_1 = nn.Sequential(CAB(n_feat * 4, n_feat * 4, kernel_size, reduction, bias=bias, act=act),
                                            CAB(n_feat * 4, n_feat * 4, kernel_size, reduction, bias=bias, act=act))

        self.feat_extract_2 = nn.Sequential(CAB(n_feat, n_feat, kernel_size, reduction, bias=bias, act=act),
                                            CAB(n_feat, n_feat, kernel_size, reduction, bias=bias, act=act))

        self.Encoder = nn.ModuleList([
            Encoder(n_feat, kernel_size, reduction, act, bias),
            Encoder(n_feat * 2, kernel_size, reduction, act, bias)
        ])

        self.Decoder = nn.ModuleList([
            Decoder(n_feat * 2, kernel_size, reduction, act, bias),
            Decoder(n_feat, kernel_size, reduction, act, bias)
        ])

        self.FAM1 = FAM(n_feat * 4)
        self.FAM2 = FAM(n_feat * 2)

        self.SAM1 = SAM(n_feat * 4, kernel_size, bias)
        self.SAM2 = SAM(n_feat * 2, kernel_size, bias)

    def forward(self, param):
        x1, x2, x4, x_2, x_4 = param

        x_en_1, x_down_2 = self.Encoder[0](x1)  # 1, 2

        z = self.FAM2(x_down_2, x2)  # 2
        x_en_2, x_down_4 = self.Encoder[1](z)  # 2, 4

        z = self.FAM1(x_down_4, x4)  # 4
        x_en_4 = self.feat_extract_1(z)  # 4

        z, x_4_out = self.SAM1(x_en_4, x_4)  # 4
        x_de_4, x_up_2 = self.Decoder[0](z, x_en_2)  # 2

        z, x_2_out = self.SAM2(x_up_2, x_2)  # 2
        x_de_2, x_up_1 = self.Decoder[1](z, x_en_1)

        x_de_1 = self.feat_extract_2(x_up_1)

        encoder_outs = [x_en_1, x_en_2, x_en_4]  # 1 2 4
        decoder_outs = [x_de_1, x_de_2, x_de_4]  # 1 2 4
        return encoder_outs, decoder_outs, x_2_out, x_4_out


##########################################################################
class MMNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=32, kernel_size=3, bias=False, reduction=4, act=nn.PReLU()):
        super(MMNet, self).__init__()

        self.shallow_feat_1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                            CAB(n_feat, n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat_2 = nn.Sequential(conv(in_c, n_feat * 2, kernel_size, bias=bias),
                                            CAB(n_feat * 2, n_feat * 2, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat_3 = nn.Sequential(conv(in_c, n_feat * 4, kernel_size, bias=bias),
                                            CAB(n_feat * 4, n_feat * 4, kernel_size, reduction, bias=bias, act=act))

        self.DF1 = DownFocus(in_c, n_feat * 2)
        self.DF2 = DownFocus(in_c, n_feat * 4)

        self.CIRNet = CIRNet(n_feat, kernel_size, reduction, bias, act)
        self.SIRNet = SIRNet(n_feat, kernel_size, reduction, bias, act)

        self.conv = nn.ModuleList([
            conv(n_feat * 4, n_feat * 2, kernel_size=1),
            conv(n_feat * 8, n_feat * 4, kernel_size=1)
        ])

        self.tail = nn.Sequential(
            CAB(n_feat * 2, n_feat * 2, kernel_size, reduction, bias=bias, act=act),
            conv(n_feat * 2, out_c, kernel_size, bias=bias))

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        x_2_focus = self.DF1(x)
        x_4_focus = self.DF2(x_2)

        x1 = self.shallow_feat_1(x)  # 1
        x2 = self.shallow_feat_2(x_2)
        x2 = self.conv[0](torch.cat((x2, x_2_focus), 1))
        x4 = self.shallow_feat_3(x_4)
        x4 = self.conv[1](torch.cat((x4, x_4_focus), 1))

        param = [x1, x2, x4, x_2, x_4]
        encoder_outs, decoder_outs, x_2_out, x_4_out = self.CIRNet(param)
        feat_sir = self.SIRNet(x1, encoder_outs, decoder_outs)
        x_1_out = torch.cat((feat_sir, decoder_outs[0]), 1)
        x_1_out = self.tail(x_1_out) + x
        return x_1_out, x_2_out, x_4_out


if __name__ == '__main__':
    x = torch.rand(1, 3, 256, 256).cuda()
    net = MMNet().cuda()
    out = net(x)
    print(out[0].shape, out[1].shape, out[2].shape)
