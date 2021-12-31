import tensorflow as tf
from tensorflow.keras.applications import VGG19
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import warnings
from tqdm import tqdm
from IPython.display import Image, HTML, display
import imageio as iio


# higher PSNR indicates better reconstruction quality, dB units
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, 1.0)

def psnr_loss(y_true, y_pred):
    return -psnr(y_true, y_pred)

# higher SSIM indicates better reconstruction quality, <-1, 1>
def ssim(y_true,y_pred):
    return tf.image.ssim(y_true, y_pred, 1.0)
    
def read_image(path_file):
    img = cv2.imread(path_file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb
    
def preprocess_image(img, img_size):
    return cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

def generator_cats_and_dogs(train_list, img_size, target_size, batch_size):
    random.shuffle(train_list)
    while True:
        for start in range(0, len(train_list), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(train_list))
                    ids_train_batch = train_list[start:end]
                    for i, ids in enumerate(ids_train_batch):
                        img_y = read_image(ids)
                        img_x = preprocess_image(img_y, img_size)
                        img_y = cv2.resize(img_y, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
                        x_batch.append(np.array(img_x,np.float32)/255.)
                        y_batch.append(np.array(img_y,np.float32)/255.)
                    x_batch = np.array(x_batch)
                    y_batch = np.array(y_batch)
                    yield x_batch,y_batch
            
            
def generator_vimeo(train_list, img_size, target_size, batch_size=1):
    random.shuffle(train_list)
    while True:
        for name in train_list:
            hr_path = os.path.join('../', 'data', 'vimeo', 'input', name.strip())
            lr_path = os.path.join('../', 'data', 'vimeo', 'low_resolution', name.strip())
            sequence = os.listdir(hr_path)
            
            x_batch = []
            y_batch = []
            for i, ids in enumerate(sequence):
                img_y = read_image(os.path.join(hr_path, ids))
                img_x = read_image(os.path.join(lr_path, ids))
                img_y = cv2.resize(img_y, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
                img_x = cv2.resize(img_x, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                
                x_batch.append(np.array(img_x, np.float32)/255.)
                y_batch.append(np.array(img_y, np.float32)/255.)
                
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            yield x_batch, y_batch
            
            
def get_stats(model, test_gen, test_list):
    mses = []
    psnrs = []
    ssims = []
    for i in tqdm(range(len(test_list))):
        x, y = next(test_gen)
        for lr, hr in zip(x, y):
            pred = model.predict(np.expand_dims(lr, axis=0))
            mses.append(np.square(pred - hr).mean(axis=None))
            psnrs.append(psnr(hr, pred))
            ssims.append(ssim(hr, pred))

    print('MSE:', np.mean(mses))
    print('PSNR:', np.mean(psnrs))
    print('SSIM:', np.mean(ssims))
    
    
def plot_comparison(model, test_gen):
    x, y = next(test_gen)
    pred = model.predict(np.expand_dims(x[0], axis=0))
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(x[0], vmin=0, vmax=1)
    axs[0].set_title('Low resolution', fontsize=24, y=1.01)
    axs[1].imshow(np.squeeze(pred).astype(np.float32), vmin=0, vmax=1)
    axs[1].set_title('Super resolution', fontsize=24, y=1.01)
    axs[2].imshow(y[0], vmin=0, vmax=1)
    axs[2].set_title('Original resolution', fontsize=24, y=1.01)

    [ax.set_axis_off() for ax in axs.ravel()]
    plt.show()
    #plt.savefig('./plots/cats.pdf', bbox_inches='tight', format='pdf')
    
def get_video_stats(model, lr, hr):
    mses = []
    psnrs = []
    ssims = []    
    for x, y in tqdm(zip(lr, hr)):
        pred = model.predict(np.expand_dims(x, axis=0))
        mses.append(np.square(pred - y).mean(axis=None))
        psnrs.append(psnr(y, pred))
        ssims.append(ssim(y, pred))
        
    print('MSE:', np.mean(mses))
    print('PSNR:', np.mean(psnrs))
    print('SSIM:', np.mean(ssims))
    
    
def plot_video_comparison(lr_path, sr_path, hr_path):
    display(HTML("<table> <tr> \
                 <td> <img src=\"" + lr_path + "\" alt=\"Drawing\" style=\"width: 300px;\"/> </td> \
                 <td> <img src=\"" + sr_path + "\" alt=\"Drawing\" style=\"width: 300px;\"/> </td> \
                 <td> <img src=\"" + hr_path + "\" alt=\"Drawing\" style=\"width: 300px;\"/> </td> \
                 </tr></table>"))
    
    
def to_0_to_255(imgs):
    return [np.uint8((np.array(img) + 1)*255/2) for img in imgs]