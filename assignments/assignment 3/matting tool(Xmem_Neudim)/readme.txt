1.Install the corresponding CUDA version of pytorch
2.pip install -r requirements.txt
3.python interactive_demo.py --images [data path] --size 1080(which can be adjusted according to your GPU memory) 
   e.g. python interactive_demo.py --images data\January\20230107_object_rem_parker\layer1  --size 1080
4.The data will be included into ./workspace and the masks will also be autoly saved in ./workspace/masks
5.Start matting manually (very similar to MiVOS)
5.Use front propagate, back propagate buttom to autoly generate masks
6.Use matting.py to mat the images (use mask and the origin images(not the downsampled images))
7.After matting, please clear the workspace.