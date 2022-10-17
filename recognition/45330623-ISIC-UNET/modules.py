import torch

def dice_coef(y_true, y_pred, smooth):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)

    return (2.0 * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

class UnetModel():
    def __init__(self):
        super(UnetModel, self).__init__()
        self.padding = 'same'

    def context_module(self, input, output_filters):
        return

    def summation():
        return

    def up_sampling():
        return

    def concatenate():
        return

    def localize():
        return
    
    def segment():
        return

    def create_pipeline():
        return