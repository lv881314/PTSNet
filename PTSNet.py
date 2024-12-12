
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from Models1.sync_batchnorm import SynchronizedBatchNorm2d
from torchvision.models import resnet34, resnet50, resnet101, resnet152, resnet18
from Models1.batchnorm import SynchronizedBatchNorm2d
from einops import rearrange, reduce, repeat, parse_shape
from Models1.utils import conv_bn_relu
from module import *



class PTSNet(nn.Module):
    def __init__(self, backbone,  sync_bn=True, pretrained=True, ResNet34M= False, criterion=nn.CrossEntropyLoss(ignore_index=255), classes = 6):
        super(PTSNet, self).__init__()
        self.ResNet34M = ResNet34M
        self.backbone = backbone
        self.criterion = criterion

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if backbone.lower() == "resnet18":
            encoder = resnet18(pretrained=pretrained)
        elif backbone.lower() == "resnet34":
            encoder = resnet34(pretrained=pretrained)
        elif backbone.lower() == "resnet50":
            encoder = resnet50(pretrained=pretrained)
        elif backbone.lower() == "resnet101":
            encoder = resnet101(pretrained=pretrained)
        elif backbone.lower() == "resnet152":
            encoder = resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError("{} Backbone not implemented".format(backbone))

        self.out_channels = [32,64,128,256,512,1024,2048]
        self.conv1_x = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.conv2_x = encoder.layer1  # 1/4
        self.conv3_x = encoder.layer2  # 1/8
        self.conv4_x = encoder.layer3  # 1/16
        self.conv5_x = encoder.layer4  # 1/32


        self.down2 = conv_block(self.out_channels[-4], self.out_channels[1], 3, 1, 1, 1, 1, bn_act=True)
        self.down3 = conv_block(self.out_channels[-3], self.out_channels[2], 3, 1, 1, 1, 1, bn_act=True)
        self.down4 = conv_block(self.out_channels[-2], self.out_channels[3], 3, 1, 1, 1, 1, bn_act=True)
        self.down5 = conv_block(self.out_channels[-1], self.out_channels[4], 3, 1, 1, 1, 1, bn_act=True)

        self.apf1 = FAM(self.out_channels[4], self.out_channels[4], self.out_channels[3], classes=classes)
        self.apf2 = FAM(self.out_channels[3], self.out_channels[3], self.out_channels[2], classes=classes)
        self.apf3 = FAM(self.out_channels[2], self.out_channels[2], self.out_channels[1], classes=classes)
        self.apf4 = FAM(self.out_channels[1], self.out_channels[1], self.out_channels[0], classes=classes)

        self.classifier = SegHead(self.out_channels[0], classes)

        self.classifier = SegHead(self.out_channels[0], classes)
        self.classifier4 = SegHead(self.out_channels[4], classes)


    def forward(self, x, y=None, z= None):
        B, C, H, W = x.size()
                x = self.conv1_x(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        if self.ResNet34M:
            x2 = self.conv2_x(x1)
        else:
            x2 = self.conv2_x(x)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        if self.backbone in ['resnet50', 'resnet101', 'resnet152']:
            x2 = self.down2(x2)
            x3 = self.down3(x3)
            x4 = self.down4(x4)
            x5 = self.down5(x5)

        CFGB = self.QKVAttention(x5)

        APF1, cls1 = self.apf1(CFGB, x5)

        APF2, cls2 = self.apf2(APF1, x4)

        APF3, cls3 = self.apf3(APF2, x3)

        APF4, cls4 = self.apf4(APF3, x2)
        
        classifier = self.classifier(APF4)
        predict = F.interpolate(classifier1, size=(H, W), mode="bilinear", align_corners=True)
        
        if self.training:
            main_loss = self.criterion(predict, y)
            return predict.max(1)[1], main_loss, main_loss
        else:
            return predict

        return predict

class LECA1(nn.Module):
    def __init__(self, channel, ratio=16):
        super(LECA1, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(channel)

    def compute_entropy(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)  # [B, C, H*W]
        norm_feature = torch.softmax(x, dim=2)
        entropy = -torch.sum(norm_feature * torch.log(norm_feature + 1e-8), dim=2)  # [B, C]
        return entropy.view(B, C, 1, 1)  # [B, C, 1, 1]

    def forward(self, x):
        B, C, H, W = x.size()
        max_value, _ = torch.max(x.view(B, C, -1), dim=2, keepdim=True)
        min_value, _ = torch.min(x.view(B, C, -1), dim=2, keepdim=True)
        diff = max_value - min_value
        norm_feature = x / (diff.view(B, C, 1, 1) + 1e-8)
        x2 = x * norm_feature

        entropy = self.compute_entropy(x2)
        tb = self.norm(entropy)
        entropy_rate_feature = self.sigmoid(tb)

        return (entropy_rate_feature * x + x)

class QKVAttention(nn.Module):
    def __init__(self, in_channels, sparsity=0.1):
        super(QKVAttention, self).__init__()
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.leca = LECA1(in_channels)
        self.sparsity = sparsity  # Fraction of values to retain in the sparse attention

    def compute_entropy(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)  # Flatten to [B, C, H*W]
        norm_feature = torch.softmax(x, dim=2)
        entropy = -torch.sum(norm_feature * torch.log(norm_feature + 1e-8), dim=2)
        return entropy.view(B, 1, H * W)  # [B, 1, H*W]

    def forward(self, x):
        B, C, H, W = x.size()

        # Compute Q, K, V
        Q = self.q_conv(x).view(B, H * W, C)  # [B, H*W, C]
        K = self.k_conv(x).view(B, C, H * W)  # [B, C, H*W]
        V = self.v_conv(x)  # [B, C, H, W]

        # Compute affinity matrix
        affinity_matrix = torch.matmul(Q, K)  # [B, H*W, H*W]

        # Apply sparsity mask
        topk = int(self.sparsity * affinity_matrix.size(1))  # Number of top elements to keep
        values, indices = torch.topk(affinity_matrix, topk, dim=-1, largest=True, sorted=False)
        mask = torch.zeros_like(affinity_matrix)
        mask.scatter_(2, indices, 1.0)  # Scatter the top k values
        sparse_affinity_matrix = affinity_matrix * mask  # Apply the mask

        # LECA1 optimization on V channel
        optimized_v = self.leca(V).view(B, C, H * W)  # [B, C, H*W]

        # Weighted sum
        weighted_sum = torch.matmul(sparse_affinity_matrix, optimized_v.permute(0, 2, 1))  # [B, H*W, C]

        # Reshape and add input
        output = weighted_sum.permute(0, 2, 1).view(B, C, H, W) + x  # [B, C, H, W]

        return output

class PyrmidFusionNet(nn.Module):
    def __init__(self, channels_high, channels_low, channel_out, classes=11):
        super(PyrmidFusionNet, self).__init__()

        self.lateral_low = conv_block(channels_low, channels_high, 1, 1, bn_act=True, padding=0)

        self.conv_low = conv_block(channels_high, channel_out, 3, 1, bn_act=True, padding=1)
        self.sa = SpatialAttention(channel_out, channel_out)

        self.conv_high = conv_block(channels_high, channel_out, 3, 1, bn_act=True, padding=1)
        self.ca = ChannelWise(channel_out)

        self.FRB = nn.Sequential(
            conv_block(2 * channels_high, channel_out, 1, 1, bn_act=True, padding=0),
            conv_block(channel_out, channel_out, 3, 1, bn_act=True, group=1, padding=1))

        self.classifier = nn.Sequential(
            conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True),
            nn.Dropout(p=0.15),
            conv_block(channel_out, classes, 1, 1, padding=0, bn_act=False))
        self.apf = conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.active = nn.Sigmoid()
        self.ESRA1 = ESRA(3*channel_out)
        self.LECA2 = LECA(channel_out)
        self.BBlock = BBlock(channel_out)
        self.ABlock = ABlock(channel_out)


    def forward(self, x_high, x_low):
        _, _, h, w = x_low.size()

        lat_low = self.lateral_low(x_low)

        high_up1 = F.interpolate(x_high, size=lat_low.size()[2:], mode='bilinear', align_corners=False)

        concate = torch.cat([lat_low, high_up1], 1)
        concate = self.FRB(concate)

        conv_high = self.conv_high(high_up1)
        conv_low = self.conv_low(lat_low)

        sa = self.sa(concate)
        ca = self.ca(concate)

        # ca = self.LECA2(concate)
        # sa = self.ESRA1(concate)

        # ca = self.BBlock(concate)
        # sa = self.ABlock(concate)

        mul1 = torch.mul(sa, conv_high)
        mul2 = torch.mul(ca, conv_low)

        # mul1 = mul1 + conv_high
        # mul2 = mul2 + conv_low
        att_out = mul1 + mul2

        # conv_highv1 = self.ESRA1(conv_high)
        # conv_lowv1 = self.ESRA1(conv_low)
        # att_outv1 = self.ESRA1(att_out)

        # cat_f = torch.cat((conv_highv1, conv_lowv1, att_outv1), dim=1)
        # b, mc, _, _ = cat_f.size()
        # act_f = self.active(torch.abs(cat_f))
        # avg_f = self.avg_pool(act_f)
        # weight = torch.split(avg_f, mc // 3, dim=1)
        # out = weight[0] * conv_high + weight[1] * conv_low + weight[2] * (att_out)

        #############
        cat_f = torch.cat((conv_high, conv_low, att_out, concate), dim=1)
        # cat_f = self.ESRA1(cat_f1)
        b, mc, _, _ = cat_f.size()
        act_f = self.active(torch.abs(cat_f))
        avg_f = self.avg_pool(act_f)
        weight = torch.split(avg_f, mc // 4, dim=1)
        out = weight[0] * conv_high + weight[1] * conv_low + weight[2] * (att_out) + weight[3] * concate
        # att_out =out

        ###原始有用####
        # cat_f = torch.cat((conv_high, conv_low, att_out), dim=1)
        # # cat_f = self.ESRA1(cat_f1)
        # b, mc, _, _ = cat_f.size()
        # act_f = self.active(torch.abs(cat_f))
        # avg_f = self.avg_pool(act_f)
        # weight = torch.split(avg_f, mc // 3, dim=1)
        # out = weight[0] * conv_high + weight[1] * conv_low + weight[2] * (att_out)
        # # att_out =out
        #####################

        sup = self.classifier(out)
        APF = self.apf(out)
        return APF,sup
