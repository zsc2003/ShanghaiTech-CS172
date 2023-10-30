from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from dataloader import SHHA_loader
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch.nn as nn

# thanks to the torch tutorial
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def data_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    data = torch.stack(data, 0)
    return [data, target]

def draw_and_save(images, coords, save_path, batch_idx):
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.485, 0.456, 0.406])
    for i in range(images.shape[0]):
        image = images[i].permute((1, 2, 0))
        image = image * std + mean
        image = image.numpy()
        coord = coords[i]
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        ax.plot(coord[:, 0], coord[:, 1], 'ro')
        plt.savefig(f"{save_path}/image_{batch_idx+i}.png")
        plt.close()

def generate_heatmap(img, ground_truth):
    # thanks to the given heatmap generator
    # https://github.com/ZhengPeng7/SANet-Keras/blob/master/generate_datasets.ipynb

    from utils import get_density_map_gaussian
    import cv2
        
    is_adaptive = False

    k = np.zeros((img.shape[0], img.shape[1]))
    # turn ground truth into numpy array
    gt = ground_truth.numpy()

    for i in range(len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    # print(gt.shape)
    # print(k.shape)
    k = get_density_map_gaussian(k, gt, adaptive_mode=is_adaptive)

    return k
    print(k)
    print(np.sum(k))

    fg, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 4))
    ax0.imshow(img)
    ax0.set_title(str(gt.shape[0]))
    ax1.imshow(np.squeeze(k), cmap=plt.cm.jet)
    ax1.set_title(str(np.sum(k)))
    plt.show()

def dice_coefficient(predicted, target):
    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum()
    dice = (2.0 * intersection) / (union + 1e-7)  # avoid division by zero
    return dice 

def draw_best(img, ground_truth, outputs):
    from utils import get_density_map_gaussian
    import cv2
        
    is_adaptive = False
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = ground_truth.numpy()

    for i in range(len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = get_density_map_gaussian(k, gt, adaptive_mode=is_adaptive)

    fg, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 4))
    ax0.imshow(img)
    ax0.set_title(str(gt.shape[0]))
    ax1.imshow(np.squeeze(k), cmap=plt.cm.jet)
    ax1.set_title(str(np.sum(k)))
    ax2.imshow(np.squeeze(outputs), cmap=plt.cm.jet)
    ax2.set_title(str(torch.sum(outputs).item()))
    plt.show()

def train(args):
    # Create Visualization folder
    import os
    if not os.path.exists("./train_images"):
        os.makedirs("./train_images")
    if not os.path.exists("./test_images"):
        os.makedirs("./test_images")
    # You can delete this part if you don't want to visualize the data

    # Define dataloader
    train_dataset = SHHA_loader(args.data_path, "train", args.output_size)
    test_dataset = SHHA_loader(args.data_path, "test", args.output_size)
    train_loader = DataLoader(
        train_dataset, args.batch_size, True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=data_collate)
    test_loader = DataLoader(
        test_dataset, args.batch_size, False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=data_collate)

    # print(len(train_dataset))
    # print(len(test_dataset))

    """
    # for batch_idx, inputs in enumerate(train_loader):
    for batch_idx, inputs in enumerate(test_loader):

        images, gt = inputs
        image = images[0].permute((1, 2, 0))
        gt = gt[0]

        heatmap = generate_heatmap(image, gt)

        print(heatmap)
        print(gt.shape)
        plt.imshow(image)
        for x,y in gt:
            plt.scatter(x,y)
        plt.show()
        return
    """

    args.batch_size = 1 # force it to be 1

    # TODO Define model and optimizer
    net = models.resnet18(weights=False)
    if args.depth == 18:
        net = models.resnet18(weights=False)
    elif args.depth == 34:
        net = models.resnet34(weights=False)
    elif args.depth == 50:
        net = models.resnet50(weights=False)
    elif args.depth == 101:
        net = models.resnet101(weights=False)
    elif args.depth == 152:
        net = models.resnet152(weights=False)

    net.fc = torch.nn.Linear(net.fc.in_features, 512 * 512) # the density map has 512*512 pixels
    net.to('cuda')

    criterion = nn.MSELoss()
    # criterion = dice_coefficient
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    iter = 0
    training_loss = []
    training_mae_loss = []
    training_mse_loss = []

    valid_losses = []
    valid_mae_losses = []
    valid_mse_losses = []

    with open(f'./info/crowd_counting_part_{args.opt}.txt', 'a') as f:
        f.write(f'=========================================================\n')
        f.write(f'start training crowd_counting with depth : {args.depth} and learning rate : {args.learning_rate}\n')
    
    for epoch in range(args.num_epoch):
        
        net.train()
        
        total_loss = 0
        total_mae = 0
        total_mse = 0

        for batch_idx, inputs in tqdm.tqdm(enumerate(train_loader)):
            images, gt = inputs
            # print(images[0].shape)
            # print(gt[0])
            # print(gt[0].shape)

            # set batch size = 1, so each batch only has one image
            image = images[0].permute((1, 2, 0))
            gt = gt[0]
            heatmap = generate_heatmap(image, gt)
            heatmap = torch.from_numpy(heatmap)
            heatmap *= 512
            heatmap = heatmap.to('cuda')
            # plot the image
            # plt.imshow(image)
            # plt.show()
            
            image, gt = image.to('cuda'), gt.to('cuda')

            # Visualize data, you can delete this part if you don't want to visualize the data
            # draw_and_save(images, gt, "./train_images", batch_idx*args.batch_size)

            # zero the parameter gradients
            optimizer.zero_grad()

            # TODO Forward
            reshaped_image = image.permute((2, 0, 1))
            # reshaped_image.reshape(1, reshaped_image.shape[0], reshaped_image.shape[1], reshaped_image.shape[2])
            # add a dimension
            reshaped_image = reshaped_image.unsqueeze(0)
            
            # print(reshaped_image.shape)
            # print("======================================")
            outputs = net(reshaped_image).reshape(512, 512)
            # make a ReLU on the output
            outputs = nn.ReLU()(outputs)
            
            # print(outputs)
            # print("-------------------------------------------------")

            # TODO Backward
            loss = criterion(outputs, heatmap)
            loss.backward()

            # TODO Update parameters
            optimizer.step()
            total_loss += loss.item()
            # print("loss = ", loss.item())

            outputs /= 512
            net_num = torch.round(torch.sum(outputs)).item()
            gt_num = gt.shape[0]
            total_mae += abs(net_num - gt_num)
            total_mse += (net_num - gt_num) ** 2

        total_loss /= len(train_dataset)
        total_mae /= len(train_dataset)
        total_mse = (total_mse / len(train_dataset)) ** 0.5

        training_loss.append(total_loss)
        training_mae_loss.append(total_mae)
        training_mse_loss.append(total_mse)
        
        net.eval()
        valid_loss = 0
        valid_mae = 0
        valid_mse = 0
        # forbidA = [10, 18, 25, 50, 65, 70, 72, 112, 124, 138, 139, 142, 147, 165, 167, 180, 182]
        # forbidB = [44, 81, 87, 88, 99]
        with torch.no_grad():
            for batch_idx, inputs in tqdm.tqdm(enumerate(test_loader)):
                # if batch_idx in forbidB:
                    # continue
                images, gt = inputs

                # set batch size = 1, so each batch only has one image
                image = images[0].permute((1, 2, 0))
                gt = gt[0]
                heatmap = generate_heatmap(image, gt)
                heatmap = torch.from_numpy(heatmap)
                heatmap *= 512

                heatmap = heatmap.to('cuda')

                # plot the image
                # plt.imshow(image)
                # plt.show()
            
                image, gt = image.to('cuda'), gt.to('cuda')

                # forward + backward + optimize
                reshaped_image = image.permute((2, 0, 1))
                reshaped_image = reshaped_image.unsqueeze(0)

                outputs = net(reshaped_image).reshape(512, 512)
                # make a ReLU on the output
                outputs = nn.ReLU()(outputs)
                
                loss = criterion(outputs, heatmap)
                
                valid_loss += loss.item()

                outputs /= 512
                net_num = torch.round(torch.sum(outputs)).item()
                gt_num = gt.shape[0]
                valid_mae += abs(net_num - gt_num)
                valid_mse += (net_num - gt_num) ** 2


        valid_loss /= len(test_dataset)
        valid_mae /= len(test_dataset)
        valid_mse = (valid_mse / len(test_dataset)) ** 0.5

        valid_losses.append(valid_loss)
        valid_mae_losses.append(valid_mae)
        valid_mse_losses.append(valid_mse)

        # TODO Print log info
        with open(f'./info/crowd_counting_part_{args.opt}.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1}, train loss: {total_loss:.4f}, train mae: {total_mae:.4f}, train mse: {total_mse:.4f}, valid loss: {valid_loss:.4f}, valid mae: {valid_mae:.4f}, valid mse: {valid_mse:.4f}\n')
        
    with open(f'./info/crowd_counting_part_{args.opt}.txt', 'a') as f:
        f.write(f'finished training crowd_counting with depth : {args.depth} and learning rate : {args.learning_rate}\n')
        f.write(f'=========================================================\n')

    # Save model checkpoints
    PATH = f'./model_para/crowd_counting/part_{args.opt}/depth_{str(args.depth)}_lr_{str(args.learning_rate)}.pth'
    torch.save(net.state_dict(), PATH)

    # turn the lists into numpy arrays and save them
    training_loss_np = np.array(training_loss)
    training_mae_loss_np = np.array(training_mae_loss)
    training_mse_loss_np = np.array(training_mse_loss)
    valid_loss_np = np.array(valid_losses)
    valid_mae_loss_np = np.array(valid_mae_losses)
    valid_mse_loss_np = np.array(valid_mse_losses)

    np.save(f'./model_para/crowd_counting/part_{args.opt}/depth_{str(args.depth)}_lr_{str(args.learning_rate)}_training_loss.npy', training_loss_np)
    np.save(f'./model_para/crowd_counting/part_{args.opt}/depth_{str(args.depth)}_lr_{str(args.learning_rate)}_training_mae_loss.npy', training_mae_loss_np)
    np.save(f'./model_para/crowd_counting/part_{args.opt}/depth_{str(args.depth)}_lr_{str(args.learning_rate)}_training_mse_loss.npy', training_mse_loss_np)
    np.save(f'./model_para/crowd_counting/part_{args.opt}/depth_{str(args.depth)}_lr_{str(args.learning_rate)}_valid_loss.npy', valid_loss_np)
    np.save(f'./model_para/crowd_counting/part_{args.opt}/depth_{str(args.depth)}_lr_{str(args.learning_rate)}_valid_mae_loss.npy', valid_mae_loss_np)
    np.save(f'./model_para/crowd_counting/part_{args.opt}/depth_{str(args.depth)}_lr_{str(args.learning_rate)}_valid_mse_loss.npy', valid_mse_loss_np)

    net.eval()
    valid_loss = 0
    valid_mae = 0
    valid_mse = 0
    
    best_loss = 1926081719491001
    best_image = 0
    best_gt = None
    best_output = None

    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            images, gt = inputs
            # Visualize data, you can delete this part if you don't want to visualize the data
            # draw_and_save(images, gt, "./test_images", batch_idx)

            # TODO Test model performance
            # set batch size = 1, so each batch only has one image

            image = images[0].permute((1, 2, 0))
            gt = gt[0]
            heatmap = generate_heatmap(image, gt)
            heatmap = torch.from_numpy(heatmap)
            heatmap *= 512

            heatmap = heatmap.to('cuda')

            # plot the image
            # plt.imshow(image)
            # plt.show()
        
            image, gt = image.to('cuda'), gt.to('cuda')

            # forward + backward + optimize
            reshaped_image = image.permute((2, 0, 1))
            reshaped_image = reshaped_image.unsqueeze(0)

            outputs = net(reshaped_image).reshape(512, 512)
            # make a ReLU on the output
            outputs = nn.ReLU()(outputs)

            loss = criterion(outputs, heatmap)
            
            valid_loss += loss.item()

            outputs /= 512
            net_num = torch.round(torch.sum(outputs)).item()
            gt_num = gt.shape[0]
            valid_mae += abs(net_num - gt_num)
            valid_mse += (net_num - gt_num) ** 2

            if loss.item() / gt_num < best_loss:
                best_loss = loss.item() / gt_num
                best_image = image
                best_gt = gt
                best_output = outputs
        

    best_output = best_output.to('cpu')
    best_image = best_image.to('cpu')
    best_gt = best_gt.to('cpu')
    draw_best(best_image, best_gt, best_output)

    valid_loss /= len(test_dataset)
    valid_mae /= len(test_dataset)
    valid_mse = (valid_mse / len(test_dataset)) ** 0.5

    with open(f'./info/crowd_counting_part_{args.opt}.txt', 'a') as f:
        f.write(f'final:  valid loss: {valid_loss:.4f}, valid mae: {valid_mae:.4f}, valid mse: {valid_mse:.4f}\n')
    
if __name__ == "__main__":
    import time
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument('--opt', type=str, default="B")
    parser.add_argument('--data_path', type=str, default="./ShanghaiTech_Crowd_Counting_Dataset/part_B_final")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--output_size', type=int, default=512)
    parser.add_argument('--depth', type=int, default=18)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    args = parser.parse_args()
    train(args)

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")