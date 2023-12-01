import matplotlib.pyplot as plt
import csv
import numpy as np
train_acc = []
with open('./log_acc_loss/train_acc.csv', 'r', newline='') as f_train_acc:
  train_acc_reader = csv.reader(f_train_acc)
  for row in train_acc_reader:
      float_row = []
      for i in row:
          float_row.append(float(i))
      train_acc.append(float_row)

eval_acc = []
with open('./log_acc_loss/eval_acc.csv', 'r', newline='') as f_eval_acc:
  eval_acc_reader = csv.reader(f_eval_acc)
  for row in eval_acc_reader:
      float_row = []
      for i in row:
          float_row.append(float(i))
      eval_acc.append(float_row)

# 绘图参数全局设置
index = range(100)
plt.rc('font',family='Times New Roman',size = 13)
# 绘制训练集acc曲线
plt.figure()
plt.plot(index,train_acc[0],label = 'Scratch')
plt.plot(index,train_acc[10],label = 'Pretrained')
plt.legend()
plt.grid(True, linestyle = "--", alpha = 0.5) #设置为虚线，透明度为0.5
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.title('Acc curve on train set')
plt.savefig('./fig_draw/scratch_pretrained_train_acc_1')
plt.show()

# 绘制验证集acc曲线
plt.figure()
plt.plot(index,eval_acc[0],label = 'Scratch')
plt.plot(index,eval_acc[10],label = 'Pretrained')
plt.legend()
plt.grid(True, linestyle = "--", alpha = 0.5) #设置为虚线，透明度为0.5
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.title('Acc curve on eval set')
plt.savefig('./fig_draw/scratch_pretrained_eval_acc_1')
plt.show()




