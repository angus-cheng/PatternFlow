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

class ContextLayer(nn.Module):
    def __init__(self, input, output_filter):
        super(ContextLayer, self).__init__()
        self.input = input
        self.output_filter = output_filter
        self.padding = 'same'
        self.context_dropout = 0.3
        self.leaky_alpha = 1e-2
        self.kernel_size = (3, 3)

        self.leaky_relu = nn.LeakyReLU(self.leaky_alpha)
        self.context_module = nn.Sequential(
            nn.InstanceNorm2d(self.input),
            self.leaky_relu,
            nn.Conv2d(in_channels=self.input, out_channels=output_filter, kernel_size=self.kernel_size, padding=self.padding),
            nn.Dropout(self.context_dropout),
            nn.InstanceNorm2d(self.input),
            self.leaky_relu,
            nn.Conv2d(in_channels=self.input, out_channels=output_filter, kernel_size=self.kernel_size, padding=self.padding),
        )

    def forward(self, x):
        x = self.context_module(x)
        return x

# class ConcatLayer(nn.Module):
#     def __init__(self, conv_layer, context_layer):
#         super(ConcatLayer, self).__init__()
#         self.conv_layer = conv_layer
#         self.context_layer = context_layer
        
#     def forward(self, x1, x2):
#         x1 = self.conv_layer(x1)
#         x2 = self.context_layer(x2)
#         x = torch.cat((x1, x2), 1)
#         return x

class LocalisationLayer(nn.Module):
    def __init__(self, input, output_filter):
        super(LocalisationLayer, self).__init__()
        self.input = input
        self.output_filter = output_filter
        self.padding = 'same'
        self.leaky_alpha = 1e-2

        self.conv1 = nn.Conv2d(input, self.output_filter, kernel_size=(3, 3), padding=self.padding)
        self.leaky_relu = nn.LeakyReLU(self.leaky_alpha)
        # self.normalize = nn.InstanceNorm2d(self.output_filter)
        self.conv2 = nn.Conv2d(self.output_filter, self.output_filter, kernel_size=(1, 1), padding=self.padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        # x = self.normalize(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        # x = self.normalize(x)
        return x


class ImprovedUnetModel(nn.Module):
    def __init__(self):
        super(ImprovedUnetModel, self).__init__()
        self.initial_output = 16
        self.leaky_alpha = 1e-2

        # Encoder: Level one
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3))
        self.leaky_relu = nn.LeakyReLU(self.leaky_alpha)
        self.context_module = ContextLayer(16, 16)

        # Encoder: Level two
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=(2, 2))
        self.context_module2 = ContextLayer(32, 32)

        # Encoder: Level three
        self.conv3 = nn.Conv2d(32, 64, (3, 3), stride=(2, 2))
        self.context_module3 = ContextLayer(64, 64)

        # Encoder: Level four
        self.conv4 = nn.Conv2d(64, 128, (3, 3), stride=(2, 2))
        self.context_module4 = ContextLayer(128, 128)

        # Encoder: Level five 
        self.conv5 = nn.Conv2d(128, 256, (3, 3), stride=(2, 2))
        self.context_module5 = ContextLayer(256, 256)

        # Upsampling layer
        self.upsample5 = nn.Upsample((63, 63), mode='bilinear')
        self.conv6 = nn.Conv2d(256, 128, (2, 2))
        # self.normalize = nn.InstanceNorm2d(128)

        # Decoder: Level four
        self.localise4 = LocalisationLayer(256, 128)
        self.upsample4 = nn.Upsample((127, 127), mode='bilinear')
        self.conv7 = nn.Conv2d(128, 128, (2, 2))
        # self.normalize = nn.InstanceNorm2d(self.initial_output * 4)

        # Decoder: Level three
        self.localise3 = LocalisationLayer(192, 64)

        # Segment 1
        self.conv8 = nn.Conv2d(64, 1, (1, 1))
        
        # Upsample
        self.upsample3 = nn.Upsample((256, 256), mode='bilinear')
        self.conv9 = nn.Conv2d(64, 16, (3, 3))

        # Decoder: Level two 
        self.localise2 = LocalisationLayer(48, 16)

        # Segment 2
        self.conv10 = nn.Conv2d(16, 1, (1, 1))

        # Upsample
        self.upsample2 = nn.Upsample((512, 512), mode='bilinear')
        self.conv11 = nn.Conv2d(16, self.initial_output, (3, 3))

        # Upsample Skip-Add 1
        self.upsample_skip_add1 = nn.Upsample((254, 254), mode='bilinear')

        # Decoder: Level one
        self.conv12 = nn.Conv2d(32, self.initial_output * 2, (3, 3))

        # Segment 3
        self.conv13 = nn.Conv2d(32, 1, (1, 1))

        # Upsample Skip-Add 2
        self.upsample_skip_add2 = nn.Upsample((508, 508), mode='bilinear')

        # Activation
        self.output = nn.Sigmoid()

    def forward(self, x):
        # Encoder: Level one
        x = x.float()
        x = self.conv1(x)
        x = self.leaky_relu(x)
        context = self.context_module(x)
        first_level = torch.add(x, context)
        first_skip = first_level

        # Encoder: Level two
        x = self.conv2(first_level)
        x = self.leaky_relu(x)
        context2 = self.context_module2(x)
        second_level = torch.add(x, context2)
        second_skip = second_level

        # Encoder: Level three
        x = self.conv3(second_level)
        x = self.leaky_relu(x)
        context3 = self.context_module3(x)
        third_level = torch.add(x, context3)
        third_skip = third_level

        # Encoder: Level four
        x = self.conv4(third_level)
        x = self.leaky_relu(x)
        context4 = self.context_module4(x)
        fourth_level = torch.add(x, context4)
        fourth_skip = fourth_level

        # Encoder: Level five
        x = self.conv5(fourth_level)
        x = self.leaky_relu(x)
        context5 = self.context_module5(x)
        fifth_level = torch.add(x, context5)
        
        # Upsampling layer
        x = self.upsample5(fifth_level)
        x = self.conv6(x)
        x = self.leaky_relu(x)
        # x = self.normalize(x)

        x = torch.cat([x, fourth_skip], dim=1)

        # Decoder: Level four
        x = self.localise4(x)
        x = self.upsample4(x)
        x = self.conv7(x)
        x = self.leaky_relu(x)
        # x = self.normalize(x)

        # Decoder: Level three
        print(x.shape)
        print(third_skip.shape)
        x = torch.cat((x, third_skip), dim=1)
        x = self.localise3(x)
        segment1 = x

        # Segmentation 1
        segment1 = self.conv8(segment1)
        segment1 = self.leaky_relu(segment1)

        # Upsampling layer
        x = self.upsample3(x)
        x = self.conv9(x)
        x = self.leaky_relu(x)
        # x = self.normalize(x)

        # Decoder: Level two
        x = torch.cat((x, second_skip), dim=1)
        x = self.localise2(x)
        segment2 = x

        # Segmentation 2
        segment2 = self.conv10(segment2)
        segment2 = self.leaky_relu(segment2)

        # Upsampling layer
        x = self.upsample2(x)
        x = self.conv11(x)
        x = self.leaky_relu(x)
        # x = self.normalize(x)

        # Skip-Add 1
        up_scaled_segment1 = self.upsample_skip_add1(segment1)
        skip_sum1 = torch.add(up_scaled_segment1, segment2)

        # Decoder: Level one
        x = torch.cat((x, first_skip), dim=1)
        x = self.conv12(x)
        x = self.leaky_relu(x)
        segment3 = x

        # Segmentation 3
        segment3 = self.conv13(segment3)
        segment3 = self.leaky_relu(segment3)

        # Skip-Add 2
        up_scaled_segment2 = self.upsample_skip_add2(segment2)
        skip_sum2 = torch.add(up_scaled_segment2, segment3)

        x = self.output(skip_sum2)

        return x 

device = "cuda"
model = ImprovedUnetModel().to(device)
# print(model)
summary(model, input_size=(3, 512, 512))