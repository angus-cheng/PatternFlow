import torch
import torchvision
import wget
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt

def dir_path_exist(path):
    path = path.replace('\\', '/')
    data_path = os.path.join(path, 'data').replace('\\', '/')

    if os.path.isdir(data_path):
        return True
    elif os.path.isdir(path):
        return True
    else: 
        os.mkdir(path)
        return False

def unzip_data(zip_path, target_dir):
    for item in os.listdir(zip_path):
        exist_dir = zip_path + '/' + item
        if os.path.isdir(exist_dir):
            return
        if item.endswith('.zip'):
            with zipfile.ZipFile(zip_path + '/' + item, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
                zip_ref.close()
    return

def download_data(web_url, path):
    dir = 'data'
    path = os.path.join(path, dir).replace('\\', '/')
    if not dir_path_exist(path):
        wget.download(web_url, out=path)
    unzip_data(path, path)

    return


class IsicImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, annotations_file=None, transform=None, target_transform=None):
        if annotations_file != None:
            self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
        # return len(self.img_dir)

    def preprocess_data(self, file):
        data = torchvision.io.read_file(file)
        images = torchvision.io.decode_jpeg(data, mode=torch.io.ImageReadMode.RGB) 
        # Normalize
        images = images / 255.0

        return images

    def preprocess_masks(self, file):
        data = torchvision.io.read_file(file)
        images = torchvision.io.decode_png(data, mode=torch.io.ImageReadMode.GRAY)
        images = images / 255.0
        # threshold 
        images = (images > 0.5).float()

        return images

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + '.jpg')
        mask_path = os.path.join(self.mask_dir, self.img_labels.iloc[idx, 0] + '_segmentation.png')

        img = torchvision.io.read_image(img_path)
        mask = torchvision.io.read_image(mask_path)

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            img = self.transform(img)
            mask = self.transfrom(mask)
        if self.target_transform:
            label = self.target_transform(label)
        
        img_new = self.preprocess_data(img)
        mask_new = self.preprocess_masks(mask)
        return img_new, mask_new

def load_data(dir, file=None):
    training_data = IsicImageDataset(img_dir=dir, annotations_file=file, transform=torchvision.transforms.Resize((512, 512)))
    return training_data

path = './'
training_data_url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip'
training_ground_truth_url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip'

download_data(training_data_url, path)
download_data(training_ground_truth_url, path)

train_dir = './data/ISIC-2017_Training_Data'
mask_dir = './data/ISIC-2017_Training_Part1_GroundTruth'
annotations_file = './data/ISIC-2017_Training_Data/ISIC-2017_Training_Data_metadata.csv'

training_data = IsicImageDataset(train_dir, mask_dir, annotations_file)
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=False)

# training_data = load_data(train_dir, train_dir + '/ISIC-2017_Training_Data_metadata.csv')
# training_ground_truth = load_data('./data/ISIC-2017_Training_Part1_GroundTruth')
# train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=False)
# train_ground_dataloader = torch.utils.data.DataLoader(training_ground_truth, batch_size=64, shuffle=False)

train_features, train_labels = next(iter(train_dataloader))
# train_ground_features, train_ground_labels = next(iter(train_ground_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# img = train_features[0].squeeze()
# img = img.T
# label = train_labels[0]
# plt.imshow(img)
# plt.show()
# print(f"Label: {label}")