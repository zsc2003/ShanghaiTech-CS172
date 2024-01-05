import argparse
import pathlib
import numpy as np
import torch.nn.functional as F
from PIL import Image
import tqdm
import cv2
import torch


# python calc_metric.py --path1 /home/cellverse/Desktop/cv/data/scenario_1/images --path2 /home/cellverse/Desktop/cv/data/scenario_1/render_ngp

def calc_PSNR(path1, path2):
    PSNR = 0
    num = 0
    
    # visit all the images in the path1
    # let img1 just be the name of the image, without the path
    for img1 in tqdm.tqdm(path1.iterdir()):
        num += 1
        img_name = img1.name
        img2 = path2 / img_name

        # calculate the PSNR between img1 and img2
        image1 = np.array(Image.open(img1))
        image2 = np.array(Image.open(img2))
        
        # if image1 is not 3-channel, then convert it to 3-channel
        if image1.shape[2] == 4:
            image1 = image1[:, :, :3]
        
        # if image2 is not 3-channel, then convert it to 3-channel
        if image2.shape[2] == 4:
            image2 = image2[:, :, :3]

        # compute MSE
        mse = np.mean((image1 - image2) ** 2)
        if mse == 0:
            PSNR += 100
        else:
            PSNR += 20 * np.log10(255.0 / np.sqrt(mse))        

    PSNR /= num
    return PSNR

def calc_SSIM(path1, path2):
    SSIM = 0
    num = 0
    
    # visit all the images in the path1
    # let img1 just be the name of the image, without the path
    for img1 in tqdm.tqdm(path1.iterdir()):
        num += 1
        img_name = img1.name
        img2 = path2 / img_name

        # calculate the SSIM between img1 and img2
        image1 = np.array(Image.open(img1))
        image2 = np.array(Image.open(img2))
        
        # if image1 is not 3-channel, then convert it to 3-channel
        if image1.shape[2] == 4:
            image1 = image1[:, :, :3]
        
        # if image2 is not 3-channel, then convert it to 3-channel
        if image2.shape[2] == 4:
            image2 = image2[:, :, :3]

        from skimage.metrics import structural_similarity
        SSIM += structural_similarity(image1, image2, channel_axis=2)

    SSIM /= num
    return SSIM

def calc_LPIPS(path1, path2):
    LPIPS = 0
    num = 0
    
    import lpips
    
    # visit all the images in the path1
    # let img1 just be the name of the image, without the path
    
    for img1 in tqdm.tqdm(path1.iterdir()):
        num += 1
        img_name = img1.name
        img2 = path2 / img_name

        # calculate the LPIPS between img1 and img2
        image1 = np.array(Image.open(img1))
        image2 = np.array(Image.open(img2))
        
        # if image1 is not 3-channel, then convert it to 3-channel
        if image1.shape[2] == 4:
            image1 = image1[:, :, :3]
        
        # if image2 is not 3-channel, then convert it to 3-channel
        if image2.shape[2] == 4:
            image2 = image2[:, :, :3]
        
        # ignore the output Loading model from:...`
        lpips_model = lpips.LPIPS(net="alex", verbose=False)
        

        # convert the png file to tensor
        image1_tensor = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float() / 255.0        
        image2_tensor = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # ignore the 4-th dimension
        image1_tensor = image1_tensor[:, :3, :, :]
        image2_tensor = image2_tensor[:, :3, :, :]

        LPIPS += lpips_model(image1_tensor, image2_tensor).item()

    LPIPS /= num
    return LPIPS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, help='The first path for images(ground truth)')
    parser.add_argument('--path2', type=str, help='The second path for images(rendered)')
    
    args = parser.parse_args()
    
    path1 = pathlib.Path(args.path1)
    path2 = pathlib.Path(args.path2)
      
    PSNR = calc_PSNR(path1, path2)
    SSIM = calc_SSIM(path1, path2)
    LPIPS = calc_LPIPS(path1, path2)
    
    # only keep 3 decimal places
    print(f'The average PSNR is {PSNR:.3f}')
    print(f'The average SSIM is {SSIM:.3f}')
    print(f'The average LPIPS is {LPIPS:.3f}')
    