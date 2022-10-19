import torch
import torchvision
import wget
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms as T

def dir_path_exist(path):
    path = path.replace('\\', '/')
    data_path = os.path.join(path, 'data').replace('\\', '/')

    if os.path.isdir(data_path) or os.path.isdir(path):
        return True
    else: 
        os.mkdir(path)
        return False


def download_data(web_url, path):
    dir = 'data'
    path = os.path.join(path, dir).replace('\\', '/')
    if not dir_path_exist(path):
        wget.download(web_url, out=path)
    unzip_data(path, path)
    return

def clean_data(path):
    for file in os.listdir(path):
        if file.endswith('superpixels.png'):
            os.remove(os.path.join(path, file))
    return

    # clean_data('./data/ISIC-2017_Training_Data')

def unzip_data(zip_path, target_dir):
    for item in os.listdir(zip_path):
        exist_dir = zip_path + '/' + item
        if os.path.isdir(exist_dir):
            return
        if item.endswith('.zip'):
            with zipfile.ZipFile(zip_path + '/' + item, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
                zip_ref.close()

        clean_data(target_dir + '/' + os.path.splitext(item)[0])
    return


def create_df(path):
    df = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name = filename.split('.')
            if name[1] != 'csv' and 'superpixels' not in name:
                df.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': df}, index = np.arange(0, len(df)))

path = './'
training_data_url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip'
training_ground_truth_url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip'

download_data(training_data_url, path)
download_data(training_ground_truth_url, path)

img_dir = './data/ISIC-2017_Training_Data'
mask_dir = './data/ISIC-2017_Training_Part1_GroundTruth'
annotations_file = './data/ISIC-2017_Training_Data/ISIC-2017_Training_Data_metadata.csv'

df = create_df(img_dir)

X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

print('Train Size   : ', len(X_train))
print('Val Size     : ', len(X_val))
print('Test Size    : ', len(X_test))

img = torchvision.io.read_image(img_dir + '/' + df['id'][100] + '.jpg')
img = T.Resize((512, 512))(img)
img = torch.transpose(img, 2, 0)
mask = torchvision.io.read_image(mask_dir + '/' + df['id'][100] + '_segmentation.png')
mask = T.Resize((512, 512))(mask)
mask = torch.transpose(mask, 2, 0) 
print('Image Size', np.asarray(img).shape)
print('Mask Size', np.asarray(mask).shape)

plt.imshow(img)
plt.imshow(mask, alpha=0.6)
plt.title('Picture with Mask Appplied')
plt.show()

class IsicDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir, mask_dir, data, transform=None, patch=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.data = data
        self.transform = transform
        self.patches = patch

        np.set_printoptions(linewidth=np.inf)
        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.img_dir + '/' + self.data[idx] + '.jpg')
        mask = torchvision.io.read_image(self.mask_dir + '/' + self.data[idx] + '_segmentation.png', torchvision.io.image.ImageReadMode.GRAY)

        if self.transform is not None:
            transformation = self.transform(img=img, mask=mask)
            img = Image.fromarray(transformation['img'])
            mask = transformation['mask']

        # if self.transform is None:
        #     img = Image.fromarray(img)

        # t = T([T.ToTensor()])
        img / 255.0
        # img = t(img)
        mask = (mask > 0.5).float()

        if self.patches: 
            img, mask = self.tiles(img, mask)

        return img, mask

    def tiles(self, img, mask):
        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768)
        img_patches = img_patches.contiguous().view(3, -1, 512, 768)
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)

        return img_patches, mask_patches

train_set = IsicDataset(img_dir, mask_dir, X_train, patch=False)
# val_set = IsicDataset(img_dir, mask_dir, X_val, patch=False)

batch_size = 3

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

print(next(iter(train_loader)))
