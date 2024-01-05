#  visit each file in current directory

import os
cwd = os.getcwd()
# for file in os.listdir(cwd):
# visit files in inverse order
for file in sorted(os.listdir(cwd), reverse=True):
    if file.endswith('.png'):
        # print(file)
        # os.rename(file, 'frame_00' + str(int(file[8:11]) + 1).zfill(3) + '.png')
        os.rename(file, 'frame_00' + str(int(file[:3]) + 1) + '.png')
