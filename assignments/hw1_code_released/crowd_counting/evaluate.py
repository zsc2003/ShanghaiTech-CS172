import matplotlib.pyplot as plt
import torch
from util.train_args import parse_args
from data.dataset import *
import os
from models.RDCNN import RDCNN
from models.CSRNet import CSRNet
from models.MCNN import MCNN

if __name__ == '__main__':
    # get ground truth heat map
    index = 10
    image_path = './ShanghaiTech_Crowd_Counting_Dataset/part_B_final/test_data/images/IMG_%s.jpg' % index
    label_path = './ShanghaiTech_Crowd_Counting_Dataset/part_B_final/test_data/ground_truth/GT_IMG_%s.mat' % index
    img = cv2.imread(image_path)  # BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB_format
    img_ori_size = np.shape(img)
    img_re_size = [math.floor(img_ori_size[0] / 4), math.floor(img_ori_size[1] / 4)]
    coords, GT_num = mat_process(label_path)
    coords_re_size = coords / [img_ori_size[0], img_ori_size[1]] * [img_re_size[0], img_re_size[1]]
    heatmap = get_heatmap(coords_re_size, img_re_size)

    # show and save the ground truth heat map
    evaluate_logs_path = './evaluate/part_B/test/MCNN'
    os.makedirs(evaluate_logs_path, exist_ok=True)
    plt.figure(dpi=300)
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')
    plt.savefig(evaluate_logs_path + '/IMG_'+str(index) + '_GT_heatmap_' + str(GT_num) + '.png')
    plt.show()

    # show and save the original image
    plt.figure()
    plt.imshow(img, cmap='jet')
    plt.axis('off')
    plt.show()
    plt.imsave(evaluate_logs_path + '/IMG_'+str(index)+'.png', img)

    # define model
    device = 'cuda:0'
    # model = RDCNN(pretained=True).to(device)
    # model = CSRNet().to(device)
    model = MCNN().to(device)

    # model load
    model_load_path = './train/checkpoints/crowd_counting_test14/history_net/history_net_95'
    state = torch.load(model_load_path, map_location=device)
    model.load_state_dict(state)

    # get the predicted heat map and number count
    transform = transforms.Compose([
        transforms.ToTensor()
        , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    pred_heatmap = model(img_tensor)
    pred_num = int(torch.sum(pred_heatmap))

    # show and save the predicted heat map
    pred_heatmap_numpy = pred_heatmap.squeeze(0).transpose(0, 2).transpose(0, 1).cpu().detach().numpy()
    plt.figure(dpi=300)
    plt.imshow(pred_heatmap_numpy, cmap='jet')
    plt.axis('off')
    plt.savefig(evaluate_logs_path + '/IMG_' + str(index) + '_pred_heatmap_' + str(pred_num) + '.png')
    plt.show()














