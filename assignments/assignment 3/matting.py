image_path = '/home/cellverse/Desktop/cv/data/scenario_3/images'
mask_path = '/home/cellverse/Desktop/cv/data/scenario_3/masks'
matting_path = '/home/cellverse/Desktop/cv/data/scenario_3/matting'

# read each image in the image_path
# let alpha be the number in the corresponding mask
# save the rgba image in the matting_path

import os
import cv2
import tqdm
for image_name in tqdm.tqdm(os.listdir(image_path)):
    img = cv2.imread(os.path.join(image_path, image_name))
    b, g, r = cv2.split(img)
    
    a = cv2.imread(os.path.join(mask_path, image_name))
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    # a.resize(r.shape)
    a = cv2.resize(a, (r.shape[1], r.shape[0]))
    a[a > 0] = 255

    rgba = cv2.merge((b, g, r, a))
    cv2.imwrite(os.path.join(matting_path, image_name), rgba)    
    # output a as a separate image