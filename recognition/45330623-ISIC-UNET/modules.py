import torch
from torch import nn

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
        self.padding = 'same'
        self.initial_output = 16
        self.context_dropout = 0.3
        self.leaky_alpha = 1e-2
        self.kernel_size = (3, 3)

        self.leaky_relu = nn.LeakyReLU(self.leaky_alpha)
        self.context_module = nn.Sequential(
            nn.InstanceNorm2d(input),
            self.leaky_relu,
            nn.Conv2d(in_channels=input, out_channels=output_filter, kernel_size=self.kernel_size, padding=self.padding),
            nn.Dropout(self.context_dropout),
            nn.InstanceNorm2d(input),
            self.leaky_relu,
            nn.Conv2d(in_channels=input, out_channels=output_filter, kernel_size=self.kernel_size, padding=self.padding),
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
    def __init__(self, input, output_filters):
        super(LocalisationLayer, self).__init__()
        self.input = input
        self.output_filters = output_filters
        self.padding = 'same'
        self.leaky_alpha = 1e-2

        self.conv1 = nn.Conv2d(input, self.output_filters, kernel_size=(3, 3), padding=self.padding)
        self.leaky_relu = nn.LeakyReLU(self.leaky_alpha)
        self.normalize = nn.LazyInstanceNorm2d()
        self.conv2 = nn.Conv2d(input, self.output_filters, kernel_size=(1, 1), padding=self.padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.normalize(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.normalize(x)
        return x


class ImprovedUnetModel(nn.Module):
    def __init__(self):
        super(ImprovedUnetModel, self).__init__()
        self.padding = 'same'
        self.initial_output = 16
        self.context_dropout = 0.3
        self.leaky_alpha = 1e-2

        # Encoder: Level one
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.initial_output, kernel_size=(3, 3), padding=self.padding)
        self.leaky_relu = nn.LeakyReLU(self.leaky_alpha)
        self.context_module = ContextLayer(self.conv1.out_channels, self.initial_output)

        # Encoder: Level two
        self.conv2 = nn.LazyConv2d(self.initial_output * 2, (3, 3), stride=(2, 2))
        self.context_module2 = ContextLayer(self.conv2.out_channels, self.initial_output * 2)

        # Encoder: Level three
        self.conv3 = nn.LazyConv2d(self.initial_output * 4, (3, 3), stride=(2, 2))
        self.context_module3 = ContextLayer(self.conv3.out_channels, self.initial_output * 4)

        # Encoder: Level four
        self.conv4 = nn.LazyConv2d(self.initial_output * 8, (3, 3), stride=(2, 2))
        self.context_module4 = ContextLayer(self.conv4.out_channels, self.initial_output * 8)

        # Encoder: Level five 
        self.conv5 = nn.LazyConv2d(self.initial_output * 16, (3, 3), stride=(2, 2))
        self.context_module5 = ContextLayer(self.conv5.out_channels, self.initial_output * 16 )

        # Upsampling layer
        self.upsample = nn.Upsample((2, 2), mode='bilinear')
        self.conv6 = nn.LazyConv2d(self.initial_output * 16, (3, 3))
        self.normalize = nn.LazyInstanceNorm2d()

        # Decoder: Level four
        self.localise4 = LocalisationLayer(self.conv6.out_channels, self.initial_output * 8)
        self.upsample = nn.Upsample((2, 2), mode='bilinear')
        self.conv7 = nn.LazyConv2d(self.initial_output * 8, (3, 3))
        self.normalize = nn.LazyInstanceNorm2d()

        # Decoder: Level three
        self.localise3 = LocalisationLayer(self.conv7.out_channels, self.initial_output * 4)

        # Segment 1
        self.conv8 = nn.LazyConv2d(1, (1, 1))
        
        # Upsample
        self.conv9 = nn.LazyConv2d(self.initial_output * 2, (3, 3))

        # Decoder: Level two 
        self.localise2 = LocalisationLayer(self.conv9.out_channels, self.initial_output * 2)

        # Segment 2
        self.conv10 = nn.LazyConv2d(1, (1, 1))

        # Upsample
        self.conv11 = nn.LazyConv2d(self.initial_output, (3, 3))

        # Decoder: Level one
        self.conv12 = nn.LazyConv2d(self.initial_output * 2, (3, 3))

        # Segment 3
        self.conv13 = nn.LazyConv2d(1, (1, 1))

        # Activation
        self.output = nn.Sigmoid()

    def forward(self, x):
        # Encoder: Level one
        x = self.conv1(x)
        x = self.leaky_relu(x)
        context = self.context_module(x, self.initial_output)
        first_level = torch.add(x, context)
        first_skip = first_level

        # Encoder: Level two
        x = self.conv2(first_level)
        x = self.leaky_relu(x)
        context2 = self.context_module2(x, self.initial_output * 2)
        second_level = torch.add(x, context2)
        second_skip = second_level

        # Encoder: Level three
        x = self.conv3(second_level)
        x = self.leaky_relu(x)
        context3 = self.context_module3(x, self.initial_output * 4)
        third_level = torch.add(x, context3)
        third_skip = third_level

        # Encoder: Level four
        x = self.conv4(third_level)
        x = self.leaky_relu(x)
        context4 = self.context_module4(x, self.initial_output * 8)
        fourth_level = torch.add(x, context4)
        fourth_skip = fourth_level

        # Encoder: Level five
        x = self.conv4(fourth_level)
        x = self.leaky_relu(x)
        context5 = self.context_module5(x, self.initial_output * 16)
        fifth_level = torch.add(x, context5)
        
        # Upsampling layer
        x = self.upsample(fifth_level)
        x = self.conv6(x)
        x = self.leaky_relu(x)
        x = self.normalize(x)

        x = torch.cat((x, fourth_skip), 1)

        # Decoder: Level four
        x = self.localise4(x)
        x = self.upsample(x)
        x = self.conv7(x)
        x = self.leaky_relu(x)
        x = self.normalize(x)

        # Decoder: Level three
        x = torch.cat((x, third_skip), 1)
        x = self.localise3(x)
        segment1 = x

        # Segmentation 1
        segment1 = self.conv8(segment1)
        segment1 = self.leaky_relu(segment1)

        # Upsampling layer
        x = self.upsample(x)
        x = self.conv9(x)
        x = self.leaky_relu(x)
        x = self.normalize(x)

        # Decoder: Level two
        x = torch.cat((x, second_skip), 1)
        x = self.localise2(x)
        segment2 = x

        # Segmentation 2
        segment2 = self.conv10(segment2)
        segment2 = self.leaky_relu(segment2)

        # Upsampling layer
        x = self.upsample(x)
        x = self.conv11(x)
        x = self.leaky_relu(x)
        x = self.normalize(x)

        # Skip-Add 1
        up_scaled_segment1 = self.upsample(segment1)
        skip_sum1 = torch.add(up_scaled_segment1, segment2)

        # Decoder: Level one
        x = torch.cat((x, first_skip), 1)
        x = self.conv12(x)
        x = self.leaky_relu(x)

        # Segmentation 3
        segment3 = self.conv13(x)

        # Skip-Add 2
        up_scaled_segment2 = self.upsample(skip_sum2)
        skip_sum2 = torch.add(up_scaled_segment2, segment3)

        x = self.output(skip_sum2)

        return x 

device = "cuda"
model = ImprovedUnetModel().to(device)
print(model)