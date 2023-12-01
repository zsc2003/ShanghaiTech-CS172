import math
import os
import torch
import datetime
import csv
import matplotlib.pyplot as plt

class trainer():
    def __init__(self, model, train_data_loader, test_data_loader, args, device):
        self.model = model
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.args = args
        self.device = device

        # interval
        self.save_interval = args.save_interval
        self.test_interval = args.test_interval

        # epoch
        self.num_epochs = args.epochs
        self.extra_epochs = args.extra_epochs
        self.start_epochs = args.start_epochs  # specify the begin epoch
        self.history_epoch = args.history_epoch  # specify the resume epoch

        # logs
        self.logs_path = os.path.join(args.logs_path, args.name)
        os.makedirs(self.logs_path, exist_ok=True)
        self.logs_fig_path = os.path.join(self.logs_path, 'log_figure')
        os.makedirs(self.logs_fig_path, exist_ok=True)
        self.checkpoints_path = os.path.join(args.checkpoints_path, args.name)
        os.makedirs(self.checkpoints_path, exist_ok=True)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)

        # loss
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

        # load weights & optimizer & iterator
        os.makedirs("%s/history_net" % (self.checkpoints_path,), exist_ok=True)
        self.latest_model_path = "%s/latest_net" % (self.checkpoints_path,)
        self.history_model_path = "%s/history_net/history_net" % (self.checkpoints_path,)
        self.optim_state_path = "%s/optim" % (self.checkpoints_path,)
        self.lrsched_state_path = "%s/lrsched" % (self.checkpoints_path,)
        self.iter_state_path = "%s/iter" % (self.checkpoints_path,)

        self.begin_epochs = self.start_epochs
        if args.resume:
            if self.history_epoch == 0:
                if os.path.exists(self.latest_model_path):
                    state = torch.load(self.latest_model_path, map_location=device)
                    self.model.load_state_dict(state)
            else:
                if os.path.exists(self.history_model_path + '_' + str(self.history_epoch)):
                    state = torch.load(self.history_model_path + '_' + str(self.history_epoch), map_location=device)
                    self.model.load_state_dict(state)
            if os.path.exists(self.optim_state_path):
                self.optimizer.load_state_dict(torch.load(self.optim_state_path, map_location=device))
            if os.path.exists(self.lrsched_state_path):
                self.lr_scheduler.load_state_dict(torch.load(self.lrsched_state_path, map_location=device))
            if self.start_epochs == 0 and os.path.exists(self.iter_state_path):
                self.begin_epochs = torch.load(self.iter_state_path, map_location=device)["iter"]

    def train_step(self, data):
        device = self.device
        img, GT_heatmap, GT_num = data['img'].to(device=device), data['heatmap'].unsqueeze(0).to(device=device), int(data['num'])
        pred_heatmap = self.model(img).to(torch.float64)
        self.optimizer.zero_grad()
        loss = self.mse_loss(pred_heatmap, GT_heatmap)
        loss.backward()
        self.optimizer.step()
        loss_dict = {}
        loss_dict['loss'] = round(loss.item(), 8)
        pred_num = int(torch.sum(pred_heatmap))
        mae = abs(GT_num-pred_num)
        mse = (GT_num-pred_num)*(GT_num-pred_num)
        error_rate =  mae/ GT_num
        acc = 1 - error_rate
        loss_dict['acc'] = round(acc, 8)
        loss_dict['mae'] = round(mae, 8)
        loss_dict['mse'] = round(mse, 8)
        return loss_dict

    def test_step(self, data, epoch, batch):
        device = self.device
        img, GT_heatmap, GT_num = data['img'].to(device=device), data['heatmap'].unsqueeze(0).to(device=device), int(data['num'])
        pred_heatmap = self.model(img)
        loss = self.mse_loss(pred_heatmap, GT_heatmap)
        loss_dict = {}
        loss_dict['loss'] = round(loss.item(), 8)
        pred_num = int(torch.sum(pred_heatmap))
        mae = abs(GT_num-pred_num)
        mse = (GT_num-pred_num)*(GT_num-pred_num)
        error_rate = mae / GT_num
        acc = 1 - error_rate
        loss_dict['acc'] = round(acc, 8)
        loss_dict['mae'] = round(mae, 8)
        loss_dict['mse'] = round(mse, 8)
        # 仅对第一个batch进行heatmap绘图
        if batch == 0:
            img_numpy = img.squeeze(0).transpose(0, 2).transpose(0, 1).cpu().detach().numpy() \
                        * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            img_numpy = img_numpy.clip(0,1)
            plt.imsave(self.logs_fig_path + '/batch0.png', img_numpy)

            GT_heatmap_numpy = GT_heatmap.squeeze(0).transpose(0, 2).transpose(0, 1).cpu().detach().numpy()
            plt.figure(dpi=300)
            plt.imshow(GT_heatmap_numpy, cmap='jet')
            plt.axis('off')
            plt.savefig(self.logs_fig_path + '/batch0_heatmap_'+str(GT_num)+'.png')
            plt.close()

            pred_heatmap_numpy = pred_heatmap.squeeze(0).transpose(0, 2).transpose(0, 1).cpu().detach().numpy()
            plt.figure(dpi=300)
            plt.imshow(pred_heatmap_numpy, cmap='jet')
            plt.axis('off')
            plt.savefig(self.logs_fig_path + '/epoch'+str(epoch)+'_'+str(pred_num)+'.png')
            plt.close()

        return loss_dict

    def start(self):

        def fmt_loss_str(losses):
            return (" " + " ".join(k + ":" + str(losses[k]) for k in losses))

        train_acc_list = []
        train_loss_list = []
        train_mae_list = []
        train_mse_list = []
        test_acc_list = []
        test_loss_list = []
        test_mae_list = []
        test_mse_list = []
        for epoch in range(self.begin_epochs, self.num_epochs + self.extra_epochs):
            now = datetime.datetime.now()
            f_train_lr = open(self.logs_path + '/train_lr.txt', mode='a')
            f_train_lr.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' lr:' +
                             str(self.optimizer.param_groups[0]["lr"]) + '\n')
            f_train_lr.close()

            print('--------Network Training--------')
            train_batch = 0
            train_acc = 0
            train_loss = 0
            train_mae = 0
            train_mse = 0
            for train_data in self.train_data_loader:
                train_losses = self.train_step(train_data)
                train_loss_str = fmt_loss_str(train_losses)
                now = datetime.datetime.now()
                f_train_ls = open(self.logs_path + '/train_logs.txt', mode='a')
                f_train_ls.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' Batch:' + str(
                    train_batch) + train_loss_str + "lr:" + str(self.optimizer.param_groups[0]["lr"]) + '\n')
                f_train_ls.close()
                print("*** train:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, "Batch:", train_batch,
                      train_loss_str, " lr:", str(self.optimizer.param_groups[0]["lr"]), '\n')
                train_batch = train_batch + 1
                # batch loss
                train_loss = train_loss + train_losses['loss']
                # batch acc
                train_acc = train_acc + train_losses['acc']
                # batch mae
                train_mae = train_mae + train_losses['mae']
                # batch mse
                train_mse = train_mse + train_losses['mse']

            # epoch loss
            train_loss= train_loss / train_batch
            train_loss_list.append(train_loss)
            # epoch acc
            train_acc = train_acc / train_batch
            train_acc_list.append(train_acc)
            # epoch mae
            train_mae = train_mae / train_batch
            train_mae_list.append(train_mae)
            # epoch mse
            train_mse = math.sqrt(train_mse / train_batch)
            train_mse_list.append(train_mse)

            now = datetime.datetime.now()
            f_train_loss = open(self.logs_path + '/train_loss.txt', mode='a')
            f_train_loss.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' train_loss:' + str(train_loss) + '\n')
            f_train_loss.close()
            f_train_acc = open(self.logs_path + '/train_acc.txt', mode='a')
            f_train_acc.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' train_acc:' + str(train_acc) + '\n')
            f_train_acc.close()
            f_train_mae = open(self.logs_path + '/train_mae.txt', mode='a')
            f_train_mae.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' train_mae:' + str(train_mae) + '\n')
            f_train_acc.close()
            f_train_mse = open(self.logs_path + '/train_mse.txt', mode='a')
            f_train_mse.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' train_mse:' + str(train_mse) + '\n')
            f_train_acc.close()

            print("*** train:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch,'train_loss:',str(train_loss),'train_acc:', str(train_acc),
                  'train_mae:', str(train_mae),'train_mse', str(train_mse),'\n')

            # network saving
            if (epoch % self.save_interval == 0) or epoch == self.num_epochs - 1:
                print('--------saving network & optimizer--------')
                torch.save(self.model.state_dict(), self.latest_model_path)
                torch.save(self.model.state_dict(), self.history_model_path+'_' + str(epoch))
                torch.save(self.optimizer.state_dict(), self.optim_state_path)
                torch.save(self.lr_scheduler.state_dict(), self.lrsched_state_path)
                torch.save({'iter':epoch+1}, self.iter_state_path)

            # network testing
            if ((epoch % self.test_interval == 0) and (epoch>0)) or epoch == self.num_epochs - 1:
                test_batch = 0
                test_acc = 0
                test_loss = 0
                test_mae = 0
                test_mse = 0
                for test_data in self.test_data_loader:
                    self.model.eval()
                    with torch.no_grad():
                        test_losses = self.test_step(test_data, epoch, test_batch)
                    self.model.train()
                    test_loss_str = fmt_loss_str(test_losses)
                    now = datetime.datetime.now()
                    f_test_ls = open(self.logs_path + '/test_logs.txt', mode='a')
                    f_test_ls.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' Batch:' + str(test_batch)
                                    + test_loss_str + '\n')
                    f_test_ls.close()
                    print("*** test:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, "Batch:", test_batch,
                            test_loss_str, '\n')
                    test_batch = test_batch + 1
                    # batch loss
                    test_loss = test_loss + test_losses['loss']
                    # batch acc
                    test_acc = test_acc + test_losses['acc']
                    # batch mae
                    test_mae = test_mae + test_losses['mae']
                    # batch mse
                    test_mse = test_mse + test_losses['mse']

                # epoch loss
                test_loss = test_loss / test_batch
                test_loss_list.append(test_loss)
                # epoch acc
                test_acc = test_acc / test_batch
                test_acc_list.append(test_acc)
                # epoch mae
                test_mae = test_mae / test_batch
                test_mae_list.append(test_mae)
                # epoch mse
                test_mse = math.sqrt(test_mse / test_batch)
                test_mse_list.append(test_mse)

                now = datetime.datetime.now()
                f_test_loss = open(self.logs_path + '/test_loss.txt', mode='a')
                f_test_loss.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' test_loss:' + str(test_loss) + '\n')
                f_test_loss.close()
                f_test_acc = open(self.logs_path + '/test_acc.txt', mode='a')
                f_test_acc.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' test_acc:' + str(test_acc) + '\n')
                f_test_acc.close()
                f_test_mae = open(self.logs_path + '/test_mae.txt', mode='a')
                f_test_mae.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' test_mae:' + str(test_mae) + '\n')
                f_test_mae.close()
                f_test_mse = open(self.logs_path + '/test_mse.txt', mode='a')
                f_test_mse.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' test_mse:' + str(test_mse) + '\n')
                f_test_mse.close()
                print("*** test:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, 'test_loss:', str(test_loss), 'test_acc:', str(test_acc),
                      'test_mae:', str(test_mae), 'test_mse', str(test_mse),'\n')

            self.lr_scheduler.step()
