import torch
from torch import nn

def dice_coef(y_true, y_pred, smooth):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

class ContextModuleLayer(nn.Module):
    def __init__(self, input, output_filter):
        super(ContextModuleLayer, self).__init__()
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

class SummationLayer(nn.Module):
    def __init__(self, conv_layer, context_layer):
        super(SummationLayer, self).__init__()
        self.conv_layer = conv_layer
        self.context_layer = context_layer
        
    def forward(self, x1, x2):
        x1 = self.conv_layer(x1)
        x2 = self.context_layer(x2)
        x = torch.cat((x1, x2), 1)
        return x

class ImprovedUnetModel(nn.Module):
    def __init__(self):
        super(ImprovedUnetModel, self).__init__()
        self.padding = 'same'
        self.initial_output = 16
        self.context_dropout = 0.3
        self.leaky_alpha = 1e-2

        # Level one
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels=self.initial_output, kernel_size=(3, 3), padding=self.padding)
        self.leaky_relu = nn.LeakyReLU(self.leaky_alpha)
        self.context_module = ContextModuleLayer(self.conv1.out_channels, self.initial_output)
        self.summation_layer = SummationLayer(self.leaky_relu, self.context_module)

        # Level two
        self.conv2 = nn.LazyConv2d(self.initial_output * 2, (3, 3), stride=(2, 2))
        self.context_module2 = ContextModuleLayer(self.conv2.out_channels, self.initial_output * 2)
        self.summation_layer2 = SummationLayer(self.leaky_relu, self.context_module2)

    def forward(self, x):
        # Level one
        x = self.conv1(x)
        x = self.leaky_relu(x)
        context = self.context_module(x, self.initial_output)
        first_level = self.summation_layer(x, context)

        # Level two
        x = self.conv2(first_level)
        x = self.leaky_relu(x)
        context2 = self.context_module2(x, self.initial_output * 2)
        second_level = self.summation_layer2(x, context2)
        return second_level

device = "cuda"
model = ImprovedUnetModel().to(device)
print(model)