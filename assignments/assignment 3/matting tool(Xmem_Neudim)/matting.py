########################### read me ###########################
#img_dir是原图的文件夹名称
#mask_dir是扣图的文件夹名称
#ans_dir是输出的文件夹名称
#shape是图片大小
###########################################################
import os
import numpy as np
import cv2
from tqdm import *
import argparse
import platform

global extensions

plat = platform.system().lower()

extensions = ['png','PNG','jpg','JPG','jpeg', 'JPEG']

def find_images(images):
    file_list = os.listdir(images)
    image_list = []
    for file_name in file_list:
        if file_name[-3:] in extensions:
            image_list.append(file_name)
        elif file_name[-4:] in extensions:
            image_list.append(file_name)
    return image_list     

parser = argparse.ArgumentParser()   
parser.add_argument('--img_dir', required = True, help = "dir of original images")
parser.add_argument('--shape', nargs = 2, required = True, help = "image size of original images, in form of W H")

args = parser.parse_args()



#img_dir = "new_images/"
img_dir = args.img_dir
mask_dir = os.path.join(img_dir, "masks")
ans_dir = os.path.join(img_dir, "output")
shape = [ int(args.shape[0]),int(args.shape[1]) ]
#shape=[2376,1584]
os.makedirs(ans_dir,exist_ok=True)
# imgs = os.listdir(img_dir)
imgs = find_images(img_dir)
masks = os.listdir(mask_dir)
imgs.sort()
masks.sort()

if(len(imgs)!=len(masks)):
    print("Please check the mask and img number rgb:{},mask:{}".format(len(imgs),len(masks)))

for i in tqdm(range(len(masks))):
    cur_img_name = img_dir + "/" + imgs[i]
    cur_mask_name = mask_dir + "/" + masks[i]

    cur_img = cv2.imread(cur_img_name) # [h,w,3]
    cur_mask = cv2.imread(cur_mask_name) # [h, w, 3]
    cur_mask[cur_mask>0] = 1 # [h, w, 3]
    #print(cur_mask)
    cur_mask = cv2.resize(cur_mask,(shape[0],shape[1])) # [h, w, 3]
    cur_mask = cur_mask[...,2] # [h, w]
    cur_mask = cur_mask.reshape(shape[1],shape[0],1) # [h, w, 1]
    cur_img = cur_img * cur_mask
    
    #print(cur_img.shape)
    #print(cur_mask.shape)
    new_image = np.concatenate([cur_img,cur_mask*255],axis=-1)
    #print(new_image.shape)
    ans_name = ans_dir + "/{:0>4d}.png".format(i) 
    cv2.imwrite(ans_name,new_image)

    
