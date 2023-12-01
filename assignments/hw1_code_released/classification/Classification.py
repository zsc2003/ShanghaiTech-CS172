import time
import csv
import torch
import torch.nn as nn
import torchvision
import sklearn.model_selection
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchstat import stat
from torchvision import models
from torchvision.transforms import transforms

test_name = 'test11'

# 加载数据
batch_size = 256
train_data_transform = transforms.Compose([
            transforms.ToTensor()
            , transforms.RandomCrop(32, padding=4)  # 先四周填充0，把图像随机裁剪成32*32
            , transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ])
test_data_transform = transforms.Compose([
            transforms.ToTensor()
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

train_data = torchvision.datasets.CIFAR10("./",train=True,download=True,transform=train_data_transform)
test_full_data = torchvision.datasets.CIFAR10("./",train=False,download=True,transform=test_data_transform)
test_data, eval_data = sklearn.model_selection.train_test_split(test_full_data, test_size=0.5)

train_loader = DataLoader(train_data,batch_size = batch_size,shuffle=True)
eval_loader = DataLoader(eval_data,batch_size = batch_size,shuffle=False)
test_loader = DataLoader(test_data,batch_size = batch_size,shuffle=False)
train_data_len = len(train_data)
eval_data_len = len(eval_data)
test_data_len = len(test_data)
print("train dataset length : ",train_data_len)
print("eval dataset length : ",eval_data_len)
print("test dataset length :",test_data_len)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 获取网络模型实例，采用cuda
## 采用resnet18
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
model.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 1x1的池化核让池化层失效
num_ftrs = model.fc.in_features  # 获取（fc）层的输入的特征数
model.fc = nn.Linear(num_ftrs, 10)
model = model.to(device)

# 分类问题采用CrossEntropyLoss
loss = nn.CrossEntropyLoss()

# optimizer 使用 Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# step decay
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# 定义训练次数
num_epoch = 100
eval_acc_best = 0.0
epoch_total_time = 0.0
train_loss_list = []
train_acc_list = []
eval_loss_list = []
eval_acc_list = []

# 开始训练
print("-----Start training-----")
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    eval_acc = 0.0
    eval_loss = 0.0
    model.train()  # 确保model是在 train model
    # 用training set进行训练
    for data in train_loader:
        optimizer.zero_grad() # 将gradient参数调0
        imgs, targets = data # 读取imgs和targets
        imgs = imgs.to(device)
        targets = targets.to(device)
        train_pred = model(imgs) #计算预测标签
        batch_loss = loss(train_pred, targets) #计算损失
        batch_loss.backward()  #利用反向传播计算gradients
        optimizer.step()  #利用gradients更新参数值
        acc = (train_pred.argmax(1) == targets).sum() #计算准确率
        train_acc = train_acc + acc  #计算总正确率
        train_loss = train_loss + batch_loss  #计算总损失
    epoch_pass_time = time.time() - epoch_start_time
    epoch_total_time = epoch_total_time + epoch_pass_time
    model.eval()  # 确保model是在evaluate model
    with torch.no_grad():
        # 用evaluate set进行模型测试，用来选择最佳模型
        for data in eval_loader:
             imgs, targets = data  # 读取imgs和targets
             imgs = imgs.to(device)
             targets = targets.to(device)
             eval_pred =  model(imgs)
             batch_loss = loss(eval_pred, targets)  # 计算损失
             acc = (eval_pred.argmax(1) == targets).sum()  # 计算准确率
             eval_acc = eval_acc + acc  # 计算总正确率
             eval_loss = eval_loss + batch_loss  # 计算总损失

    train_loss = train_loss/train_data_len
    train_acc = train_acc/train_data_len
    eval_loss = eval_loss/eval_data_len
    eval_acc = eval_acc/eval_data_len

    train_loss_list.append(train_loss.cpu().detach().numpy())
    train_acc_list.append(train_acc.cpu().detach().numpy())
    eval_loss_list.append(eval_loss.cpu().detach().numpy())
    eval_acc_list.append(eval_acc.cpu().detach().numpy())

    print('[%03d/%03d] %2.6f (s) for training | Train Acc: %3.6f Train Loss: %3.10f '
          'Learning Rate: %3.6f | Eval Acc: %3.6f Eval Loss: %3.10f ' %
          (epoch + 1, num_epoch, epoch_pass_time, train_acc, train_loss, optimizer.param_groups[0]["lr"], eval_acc, eval_loss))

    if eval_acc > eval_acc_best:
        eval_acc_best = eval_acc
        torch.save(model, "model_best.pth")
        print("-----The best model until now has been saved-----")

    lr_scheduler.step()

# 记录acc与loss数据
with open('./log_acc_loss/train_acc.csv', 'a+', newline='') as f_train_acc:
  writer_train_acc = csv.writer(f_train_acc)
  writer_train_acc.writerow(train_acc_list)
with open('./log_acc_loss/train_loss.csv', 'a+', newline='') as f_train_loss:
  writer_train_loss = csv.writer(f_train_loss)
  writer_train_loss.writerow(train_loss_list)
with open('./log_acc_loss/eval_acc.csv', 'a+', newline='') as f_eval_acc:
  writer_eval_acc = csv.writer(f_eval_acc)
  writer_eval_acc.writerow(eval_acc_list)
with open('./log_acc_loss/eval_loss.csv', 'a+', newline='') as f_eval_loss:
  writer_eval_loss = csv.writer(f_eval_loss)
  writer_eval_loss.writerow(eval_loss_list)

print("-----Training stage has been finished-----")
model_best = torch.load("model_best.pth")
print("-----Start the testing stage using the best model-----")
test_acc = 0.0
test_loss = 0.0
model_best.eval()  # 确保model是在evaluate model
# 开始最终测试
test_begin_time = time.time()
with torch.no_grad():
    for data in test_loader:
         imgs, targets = data  # 读取imgs和targets
         imgs = imgs.to(device)
         targets = targets.to(device)
         test_pred =  model_best(imgs)
         batch_loss = loss(test_pred, targets)  # 计算损失
         acc = (test_pred.argmax(1) == targets).sum()  # 计算准确率
         test_acc = test_acc + acc  # 计算总正确率
         test_loss = test_loss + batch_loss  # 计算总损失

test_loss = test_loss/test_data_len
test_acc = test_acc/test_data_len
test_time = time.time()-test_begin_time
print('%2.6f (s) for total train | %2.6f (s) for avg train | %2.6f (s) for total test | %2.10f (s) for avg test | Test Acc: %3.6f Test Loss: %3.10f'
               %(epoch_total_time, epoch_total_time/num_epoch, test_time, test_time/test_data_len, test_acc, test_loss))
with open('./log_acc_loss/test_acc.csv', 'a+', newline='') as f_test_acc:
  writer_test_acc = csv.writer(f_test_acc)
  writer_test_acc.writerow([test_acc.cpu().detach().numpy()])
with open('./log_acc_loss/test_loss.csv', 'a+', newline='') as f_test_loss:
  writer_test_loss = csv.writer(f_test_loss)
  writer_test_loss.writerow([test_loss.cpu().detach().numpy()])


# 绘图参数全局设置
index = range(num_epoch)
plt.rc('font',family='Times New Roman',size = 13)
## 绘制验证集与测试集上的loss, acc曲线
# 画Loss
plt.figure()
plt.plot(index,train_loss_list,label = 'train loss')
plt.plot(index,eval_loss_list,label = 'eval loss')
plt.legend()
plt.grid(True, linestyle = "--", alpha = 0.5) #设置为虚线，透明度为0.5
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss curve on train set and eval set')
plt.savefig('./log_figure/'+test_name+'_LossCurve')
plt.show()

# 画Acc
plt.figure()
plt.plot(index,train_acc_list,label='train acc')
plt.plot(index,eval_acc_list,label='eval acc')
plt.legend()
plt.grid(True, linestyle = "--", alpha = 0.5) #设置为虚线，透明度为0.5
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Acc curve on train set and eval set')
plt.savefig('./log_figure/'+test_name+'_AccCurve')
plt.show()
