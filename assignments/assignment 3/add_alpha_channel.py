import cv2
import numpy as np
for i in range(1,270):
    img = cv2.imread("./images/frame_{}.png".format(str(i).zfill(5)))
    # add an alpha channel to the image
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 #creating a dummy alpha channel image.
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    import os
    
    os.makedirs("image_new", exist_ok=True)
    cv2.imwrite("image_new/frame_{}.png".format(str(i).zfill(5)), img_BGRA)