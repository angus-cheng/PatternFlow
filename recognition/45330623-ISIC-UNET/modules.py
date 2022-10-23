import torch
from torch import nn
from torchsummary import summary

def dice_coef(y_true, y_pred, smooth):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

class ContextModule(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(ContextModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = ContextModule(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2), 
            ContextModule(in_ch, out_ch),
            nn.LeakyReLU(inplace=False))

        self.context = nn.Sequential(
            nn.MaxPool2d(2),
            ContextModule(in_ch, out_ch)
        )
    def forward(self, x):
        x1 = self.mpconv(x)
        x2 = self.context(x)
        x = torch.add(x1, x2)
        return x

class LocalisationModule(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(LocalisationModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=False),
            nn.BatchNorm2d(out_ch),
            # nn.Conv2d(out_ch, out_ch, (1, 1), padding=1),
            nn.LeakyReLU(inplace=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class SegmentationLayer(nn.Module):

    def __init__(self, in_ch):
        super(SegmentationLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 1, 1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.Upsample((out_ch, out_ch), mode="bilinear", align_corners=True)

        self.conv = LocalisationModule(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ImprovedUNetModel(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ImprovedUNetModel, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.seg1 = SegmentationLayer(128)
        self.upsample_seg1 = nn.Upsample((256, 256), mode='bilinear', align_corners=True)
        self.up3 = up(256, 64)
        self.seg2 = SegmentationLayer(64)
        self.upsample_seg2 = nn.Upsample((256, 256), mode='bilinear', align_corners=True)
        self.up4 = up(128, 64)
        self.seg3 = SegmentationLayer(64)
        self.upsample_seg3 = nn.Upsample((256, 256), mode='bilinear', align_corners=True)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        seg1 = self.seg1(x)
        seg1 = self.upsample_seg1(seg1)
        x = self.up3(x, x2)
        seg2 = self.seg2(x)
        seg2 = self.upsample_seg2(seg2)
        seg2 = torch.add(seg1, seg2)
        x = self.up4(x, x1)
        seg3 = self.seg3(x)
        seg3 = self.upsample_seg3(seg3)
        seg3 = torch.add(seg2, seg3)
        x = self.outc(x)
        return torch.sigmoid(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ImprovedUNetModel(3, 4).to(device)
summary(model, input_size=(3, 512, 512))