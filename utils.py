from __future__ import division
from __future__ import print_function

import numpy as np
import csv
from scipy import misc

from  bird_dataset_generator import BirdClassificationGenerator


#pre-processing stuff
def gray2rgb(img, path):
    if len(img.shape) < 3:
        img = np.stack((img,)*3,axis=2)
    return img

def random_flip_lr(img):
    rand_num = np.random.rand(1)
    if rand_num > 0.5:
        img = np.flip(img, 1)
    return img

def random_brightness(img):
    rand_num = np.random.randint(3, high=10, size=1)/10.0
    img = img * rand_num;
    img = img.astype(dtype=np.uint8)
    return img


def normalize_input(img, height):
    img = img.astype(dtype=np.float32)
    img[:,:,0] -= 103.939
    img[:,:,1] -= 116.779
    img[:,:,2] -= 123.68
    #img = np.divide(img, 255.0)
    return img 

def add_random_noise(img):
    return img + np.random.normal(0, 50.0, (img.shape))

def preprocess_image(img, height, width, set_type):
    img = misc.imresize(np.asarray(img), (height, width))
    if set_type == 'train':   
        img = random_flip_lr(img)
        img = random_brightness(img)
    img = normalize_input(img, height)
    #img = add_random_noise(img)
    return img



# return a batch of input(images, labels) to feed into plavceholders
def get_batch(generator_type, set_type, height, width):
    imgs = []
    if set_type == 'train' or set_type == 'val':
        for paths, bbs, labels in generator_type:
            for i  in range(len(paths)):
                img = gray2rgb(misc.imread(paths[i]), paths[i])
                img = img[bbs[i][1]:bbs[i][1]+bbs[i][3], bbs[i][0]:bbs[i][0]+bbs[i][2],:]
                img = preprocess_image(img, height, width, set_type)
                imgs.append(img)
            imgs = np.asarray(imgs)
            break
        return imgs, labels
    else:
        for paths, bbs in generator_type:
            for i  in range(len(paths)):
                img = gray2rgb(misc.imread(paths[i]), paths[i])
                img = img[bbs[i][1]:bbs[i][1]+bbs[i][3], bbs[i][0]:bbs[i][0]+bbs[i][2],:]
                imgs.append(preprocess_image(img, height, width, set_type))
            imgs = np.asarray(imgs)
            break
        return imgs, None



#store in required csv format
def save_csv(model_pred, obj):
    with open('submission.csv',"w") as f:
        writer = csv.writer(f, delimiter=',',  quotechar='"', quoting=csv.QUOTE_ALL)
        row = ['Id', 'Category']
        writer.writerow(row)
        for i in range(len(obj.test_list)):
            row = []
            row.append(obj.test_list[i]) 
            row.append(model_pred[i]+1)
            writer.writerow(row)
