import torch
import torchvision
import wget
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

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

def create_df(path):
    df = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.split('.')[1] != 'csv':
                df.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': df}, index = np.arange(0, len(df)))

path = './'
training_data_url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip'
training_ground_truth_url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip'

download_data(training_data_url, path)
download_data(training_ground_truth_url, path)

image_dir = './data/ISIC-2017_Training_Data'
mask_dir = './data/ISIC-2017_Training_Part1_GroundTruth'
annotations_file = './data/ISIC-2017_Training_Data/ISIC-2017_Training_Data_metadata.csv'

df = create_df(image_dir)
print(len(df))

X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

print('Train Size   : ', len(X_train))
print('Val Size     : ', len(X_val))
print('Test Size    : ', len(X_test))

img = torchvision.io.read_image(image_dir + '/' + df['id'][100] + '.jpg')
img = torchvision.transforms.Resize((512, 512))(img)
img = torch.transpose(img, 2, 0)
mask = torchvision.io.read_image(mask_dir + '/' + df['id'][100] + '_segmentation.png')
mask = torchvision.transforms.Resize((512, 512))(mask)
mask = torch.transpose(mask, 2, 0) 
print('Image Size', np.asarray(img).shape)
print('Mask Size', np.asarray(mask).shape)

plt.imshow(img)
plt.imshow(mask, alpha=0.6)
plt.title('Picture with Mask Appplied')
plt.show()