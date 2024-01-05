import os
from os import path
import shutil
import collections

import cv2
from PIL import Image
if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image
import numpy as np

from util.palette import davis_palette
import progressbar
 
import imagesize

# https://bugs.python.org/issue28178
# ah python ah why

def find_images(images):
    extensions = ['png','PNG','jpg','JPG']
    print(images)
    file_list = os.listdir(images)
    image_list = []
    for file_name in file_list:
        if file_name[-3:] in extensions:
            image_list.append(os.path.join(images, file_name))
    image_list, (w, h) = check_images(image_list)
    return image_list, w, h        
        
def check_images(image_list):
    size_dict = {}
    # for index, image_path in enumerate(image_list):
    #     w, h = imagesize.get(image_path)
    #     if size_dict.get((w, h),0):
    #         size_dict[(w, h)].append(index)
    #     else:
    #         size_dict[(w, h)] = [index]
    for image_path in image_list:
        w, h = imagesize.get(image_path)
        if size_dict.get((w, h), 0):
            size_dict[(w, h)] += 1
        else:
            size_dict[(w, h)] = 1
    if len(size_dict) == 1:
        print("All images are checked for the same size")
        return image_list, (w, h)
    else:
        size_kept = max(size_dict,key=lambda x:size_dict[x])
        print("%d kind of size found, keep size with the largest number: " %(len(size_dict)), size_kept)
        image_list_new = []
        for image_path in image_list:
            w, h = imagesize.get(image_path)
            if (w, h) == size_kept:
                image_list_new.append(image_path)
        return image_list_new, size_kept


class LRU:
    def __init__(self, func, maxsize=128):
        self.cache = collections.OrderedDict()
        self.func = func
        self.maxsize = maxsize
 
    def __call__(self, *args):
        cache = self.cache
        if args in cache:
            cache.move_to_end(args)
            return cache[args]
        result = self.func(*args)
        cache[args] = result
        if len(cache) > self.maxsize:
            cache.popitem(last=False)
        return result

    def invalidate(self, key):
        self.cache.pop(key, None)


class ResourceManager:
    def __init__(self, config):
        # determine inputs
        images = config['images']
        video = config['video']
        self.workspace = config['workspace']
        self.size = config['size']
        self.palette = davis_palette
        
        self.export_in_situ = config['dont_export_in_situ']
        self.matting = config['no_matting']

        # create temporary workspace if not specified
        if self.workspace is None:
            if images is not None:
                basename = path.basename(images)
            elif video is not None:
                basename = path.basename(video)[:-4]
            else:
                raise NotImplementedError(
                    'Either images, video, or workspace has to be specified')

            self.workspace = path.join('./workspace', basename)

        if self.export_in_situ:
            self.workspace = images

        print(f'Workspace is in: {self.workspace}')
        
        print("Export in original folder is: %s" %(self.export_in_situ))

        # determine the location of input images
        need_decoding = False
        need_resizing = False
        if path.exists(path.join(self.workspace, 'images')):
            pass
        elif images is not None:
            need_resizing = True
        elif video is not None:
            # will decode video into frames later
            need_decoding = True

        # create workspace subdirectories
        self.image_dir = path.join(self.workspace, 'images')
        self.mask_dir = path.join(self.workspace, 'masks')
        self.matting_dir = path.join(self.workspace, 'matting')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.matting_dir, exist_ok=True)

        # convert read functions to be buffered
        self.get_image = LRU(self._get_image_unbuffered, maxsize=config['buffer_size'])
        self.get_mask = LRU(self._get_mask_unbuffered, maxsize=config['buffer_size'])

        # extract frames from video
        if need_decoding:
            self._extract_frames(video)

        # copy/resize existing images to the workspace
        if need_resizing:
            self._copy_resize_frames(images)
        else:
            self.image_list, self.w_o,self.h_o = find_images(images)
        # read all frame names
        self.names = sorted(os.listdir(self.image_dir))
        self.names = [f[:-4] for f in self.names] # remove extensions
        self.length = len(self.names)

        assert self.length > 0, f'No images found! Check {self.workspace}/images. Remove folder if necessary.'

        print(f'{self.length} images found.')

        self.height, self.width = self.get_image(0).shape[:2]
        self.visualization_init = False

    def _extract_frames(self, video):
        cap = cv2.VideoCapture(video)
        frame_index = 0
        print(f'Extracting frames from {video} into {self.image_dir}...')
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        while(cap.isOpened()):
            _, frame = cap.read()
            if frame is None:
                break
            if self.size > 0:
                h, w = frame.shape[:2]
                new_w = (w*self.size//min(w, h))
                new_h = (h*self.size//min(w, h))
                if new_w != w or new_h != h:
                    frame = cv2.resize(frame,dsize=(new_w,new_h),interpolation=cv2.INTER_AREA)
            cv2.imwrite(path.join(self.image_dir, f'{frame_index:07d}.jpg'), frame)
            frame_index += 1
            bar.update(frame_index)
        bar.finish()
        print('Done!')

    def _copy_resize_frames(self, images):
        #image_list = os.listdir(images)
        self.image_list, self.w_o, self.h_o= find_images(images)
        print(f'Copying/resizing frames into {self.image_dir}...')
        for image_name in progressbar.progressbar(self.image_list):
            if self.size < 0:
                # just copy
                shutil.copy2(path.join(images, image_name), self.image_dir)
            else:
                # frame = cv2.imread(path.join(images, image_name))
                frame = cv2.imread(image_name)
                h, w = frame.shape[:2]
                new_w = (w*self.size//min(w, h))
                new_h = (h*self.size//min(w, h))
                if new_w != w or new_h != h:
                    frame = cv2.resize(frame,dsize=(new_w,new_h),interpolation=cv2.INTER_AREA)
                cv2.imwrite(path.join(self.image_dir, os.path.basename(image_name)), frame)

        print('Done!')

    def save_mask(self, ti, mask):
        # mask should be uint8 H*W without channels
        assert 0 <= ti < self.length
        assert isinstance(mask, np.ndarray)

        
        if self.matting:
            self.save_matting(ti, mask)
            

        mask = Image.fromarray(mask)
        mask.putpalette(self.palette)
        mask.save(path.join(self.mask_dir, self.names[ti]+'.png'))

        

        self.invalidate(ti)
    
    def save_matting(self, ti, cur_mask):
        cur_img_name = self.image_list[ti]
        cur_img = cv2.imread(cur_img_name)
        
        
        cur_mask[cur_mask>0] = 1
        

        cur_mask = cv2.resize(cur_mask,(self.w_o,self.h_o))
        cur_mask = np.expand_dims(cur_mask, axis = -1)
        cur_img = cur_img * cur_mask

        new_image = np.concatenate([cur_img, cur_mask * 255], axis = -1)
        ans_name = os.path.join(self.matting_dir, self.names[ti] + '.png')
        cv2.imwrite(ans_name, new_image)

    def save_visualization(self, ti, image):
        # image should be uint8 3*H*W
        assert 0 <= ti < self.length
        assert isinstance(image, np.ndarray)
        if not self.visualization_init:
            self.visualization_dir = path.join(self.workspace, 'visualization')
            os.makedirs(self.visualization_dir, exist_ok=True)
            self.visualization_init = True

        image = Image.fromarray(image)
        image.save(path.join(self.visualization_dir, self.names[ti]+'.jpg'))

    def _get_image_unbuffered(self, ti):
        # returns H*W*3 uint8 array
        assert 0 <= ti < self.length

        try:
            image = Image.open(path.join(self.image_dir, self.names[ti]+'.jpg'))
        except FileNotFoundError:
            image = Image.open(path.join(self.image_dir, self.names[ti]+'.png'))
        image = np.array(image)
        return image

    def _get_mask_unbuffered(self, ti):
        # returns H*W uint8 array
        assert 0 <= ti < self.length

        mask_path = path.join(self.mask_dir, self.names[ti]+'.png')
        if path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = np.array(mask)
            return mask
        else:
            return None

    def read_external_image(self, file_name, size=None):
        image = Image.open(file_name)
        is_mask = image.mode in ['L', 'P']
        if size is not None:
            # PIL uses (width, height)
            image = image.resize((size[1], size[0]), 
                    resample=Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC)
        image = np.array(image)
        return image

    def invalidate(self, ti):
        # the image buffer is never invalidated
        self.get_mask.invalidate((ti,))

    def __len__(self):
        return self.length

    @property
    def h(self):
        return self.height

    @property
    def w(self):
        return self.width
