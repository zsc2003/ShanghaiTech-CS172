import torch
import os
import glob
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from torchvision.transforms import transforms
import math

def mat_process(label_path):
    data = sio.loadmat(label_path)
    coords = data['image_info'][0][0][0][0][0]
    num = int(data['image_info'][0][0][0][0][1][0])
    return coords, num

def img_label_show(img_path, label_path):
    # 读取照片
    img = cv2.imread(img_path)  # BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB_format
    img_ori_size = np.shape(img)

    # original size
    plt.figure()
    plt.imshow(img)
    # 读取坐标和人数
    coords, num = mat_process(label_path)
    # 画坐标点
    plt.scatter(coords[:,0], coords[:,1], color='r', marker='o', edgecolors='w', s=20)
    plt.axis('off')
    plt.show()

    # 1/4 of original size
    img_resize = cv2.resize(img,dsize=(math.floor(img_ori_size[1]/8), math.floor(img_ori_size[0]/8)))
    img_re_size = np.shape(img_resize)
    coords_re_size = coords/[img_ori_size[0], img_ori_size[1]] * [img_re_size[0], img_re_size[1]]
    plt.figure()
    plt.imshow(img_resize)
    # 画坐标点
    plt.scatter(coords_re_size[:,0], coords_re_size[:,1], color='r', marker='o', edgecolors='w', s=20)
    plt.axis('off')
    plt.show()

    # heatmap of 1/4 original size
    heatmap = get_heatmap(coords_re_size, [img_re_size[0],img_re_size[1]])
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')
    plt.show()
    print(int(np.sum(heatmap)),num)

def calc_d_avg(coords, k):
    d_avg_list = []
    for i , coor in enumerate(coords):
        # 为每一个坐标点计算与其他坐标之间的距离
        dis = np.linalg.norm(coords - coor, axis=1)
        # 将距离从小到大排序
        dis.sort()
        # 取前k+1个, 第一个一定是0, 即自己本身
        if k+1 <= len(dis):
            d_avg = np.sum(dis[:k+1])/k
        else:
            d_avg = np.sum(dis) / (len(dis)-1)
        d_avg_list.append(d_avg)
    return np.array(d_avg_list)


def get_heatmap(coords, size, beta=0.3, k=10):

    # 为每个坐标计算d_avg
    d_avg = calc_d_avg(coords, k)

    # 创建heatmap
    heatmap = np.zeros(size)

    # 为每个坐标构建heatmap并累加
    for i, coor in enumerate(coords):
        try:
            # 创建当前坐标的
            cur_coor_heatmap = np.zeros(size)
            x, y = coor
            x, y = int(x), int(y)
            # 得到冲激函数
            cur_coor_heatmap[y][x] = 1
            # 根据公式 sigma = beta * d
            sigma = beta * d_avg[i]
            # 用高斯模板进行卷积 得到当前坐标的heatmap
            cur_coor_heatmap = cv2.GaussianBlur(cur_coor_heatmap, (3,3), sigma)
            # 对heatmap进行累加
            heatmap += cur_coor_heatmap
        except IndexError:
            continue

    return heatmap


class crowddataset(torch.utils.data.Dataset):

    def __init__(self, path, stage='train'):
        super(crowddataset, self).__init__()
        self.base_path = os.path.join(path,stage+'_data')
        print('loading dataset:', path, stage)
        self.image_path = sorted(glob.glob(os.path.join(self.base_path, 'images', '*.jpg'),recursive=True))
        self.transform = transforms.Compose([
            transforms.ToTensor()
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index, ):
        img_path = self.image_path[index]
        img = cv2.imread(img_path)  # BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB_format
        img_ori_size = np.shape(img)  # image original size

        img = self.transform(img)
        img_basename = os.path.basename(img_path)[:-4]
        label_path = os.path.join(self.base_path, 'ground_truth', 'GT_'+img_basename+'.mat')
        coords, num = mat_process(label_path)
        img_re_size = [math.floor(img_ori_size[0]/8), math.floor(img_ori_size[1]/8)]
        coords_re_size = coords / [img_ori_size[0], img_ori_size[1]] * [img_re_size[0], img_re_size[1]]
        heatmap = get_heatmap(coords_re_size, img_re_size)
        result = {
            'img': img,
            'heatmap': heatmap,
            'num': num,
        }
        return result

if __name__ == '__main__':
    index = 250
    img_path = '../ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images/IMG_%s.jpg' % index
    label_path = '../ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/ground_truth/GT_IMG_%s.mat' % index
    img_label_show(img_path, label_path)


    # train_dataset = crowddataset('../ShanghaiTech_Crowd_Counting_Dataset/part_A_final', stage="train")
    # train_data_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=1,
    #     shuffle = False,
    # )
    # for train_data in train_data_loader:
    #     img, GT_heat, num = train_data['img'], train_data['heatmap'], train_data['num']
    #     print('hello')