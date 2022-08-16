"""模拟终端，每个终端为一个进程"""

import multiprocessing as mp
import pickle
import random
from multiprocessing import connection

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import network
from comm import get_msg, recv_large_obj, send_large_obj
from dataset import UCF101Dataset, VideoLoader
import os.path as osp


def inf_generator(data_loader):
    """封装 data_loader 对象，让他无限循环"""
    while True:
        for data_batch in data_loader:
            yield data_batch


class Client(mp.Process):
    """client 进程对象

    对象继承自 mp.Process，该对象为多进程时，一个进程的抽象。
    外部创建一个 Client 对象 client 后，调用 client.start()，会拉起一个子进程执行 client.main 函数
    """
    def __init__(self, client_idx, ps_conn, cl_conn, trainset_index, *,
                 batch_size, lr, tensorboard_fd):
        """初始化一个 client 进程对象

        Args:
            client_idx: 服务器的编号，用于区分服务器
            cl_conn (connection.Connection): 与参数服务器（主进程）通讯的 Pipe 端，客户端进程使用该对象发送和接受消息
            ps_conn (connection.Connection): 与本进程通讯的 Pipe 端，主进程使用该对象发送和接受消息
            trainset_index: 存储训练数据索引的文件路径，对应该终端的数据集
            tensorboard_fd: 存储日志 tensorboard 文件的文件夹

        Notes:
            - 初始化对象后，并不会拉起子进程，只会进行一些参数配置，需要调用 client.start() 方法才会启动子进程
            - Pipe 的两端都传入并作为了 Client 的一个属性，这是为了方便调用，实际上要注意，主进程只使用 ps_conn（在 server.py 中），
              客户进程只使用 cl_conn（在 Client.main 中）
        """
        self.server_idx = client_idx
        self.ps_conn = ps_conn
        self.cl_conn = cl_conn
        self.trainset_index = trainset_index
        self.tensorboard_fd = tensorboard_fd

        self.batch_size = batch_size
        self.lr = lr
        super().__init__(target=self.main)

    def main(self):
        """client 子进程的主函数，每个被模拟的 client 会执行该函数"""
        print("client {} 开始运行".format(self.server_idx))

        # 随机种子指定
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        # 该 client 进程负责的数据集
        trainset = UCF101Dataset(video_index=self.trainset_index,
                                 video_processor=VideoLoader(train=True),
                                 selected_classes=config.SELECTED_CLASSES)
        # 该对象基于传入的 dataset 完成 batch 拼接操作
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        # 原始的 loader 过完一遍数据集就停止了，这里让封装一个无限循环的对象
        inf_data_iter = iter(inf_generator(train_loader))

        # 创建模型
        model = network.C3D(num_classes=config.NUM_CLASSES)
        # 加载预训练模型
        model.load_pretrained_weights(osp.join(config.RES_FD, 'ucf101-caffe.pth'))
        model.to(torch.device('cpu'))  # 为了节省显存，先放在 CPU

        # 训练初始化（损失函数等）
        criterion = nn.CrossEntropyLoss().to(config.DEVICE)
        train_params = [{'params': model.get_base_params(), 'lr': self.lr},
                        {'params': model.get_top_params(), 'lr': self.lr * 10}]
        # 创建优化器
        #   train_params 传入待优化的参数，底层学习率低顶层学习率高，因此传入一个字典的数组，来指定各层的学习率
        #   lr 传入默认的学习率，由于上面指定了每一层的学习率，理论上不需要传入
        #   momentum 为动量，可以参考 momentum SGD 算法
        #   weight_decay 为正则化参数，可以参考 L2 正则化
        optimizer = optim.SGD(train_params, lr=self.lr, momentum=0.9, weight_decay=5e-4)

        while True:
            # 本轮轮次编号
            round_idx, = get_msg(self.cl_conn, config.OP_GET_ROUND)
            # -1 表示训练结束，进程关闭
            if round_idx == -1:
                return
            # 本轮迭代，本地迭代次数
            num_local_steps, = get_msg(self.cl_conn, config.OP_GET_NSTEP)
            print('CLIENT {} received round_idx {} and start, running for {} iterations'.format(
                self.server_idx, round_idx, num_local_steps))

            print('CLIENT {} start to recv the current model from server...'.format(self.server_idx))
            # 从 server 接受上一轮结束时的模型参数
            global_model_state = recv_large_obj(self.cl_conn, start_tag=config.OP_TRANSFER_MODEL_START,
                                                end_tag=config.OP_TRANSFER_MODEL_END)
            print('CLIENT {} received the model, start local training'.format(self.server_idx))

            # 加载模型参数
            model.load_state_dict(global_model_state)

            # 统计训练过程中的损失函数和正确率
            running_loss = 0.0
            running_corrects = 0.0
            total_cnt = 0

            # 开始训练，模型放到 GPU
            model.train()
            model.to(config.DEVICE)

            # 迭代 num_local_steps 次
            for _ in range(num_local_steps):
                inputs, labels = next(inf_data_iter)
                # 数据传入对应设备
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                # 如果不执行这一句，模型的梯度就会不断累积，与预期不符，因此 pytorch 中经典的做法是，每一迭代都清零梯度
                optimizer.zero_grad()

                # 模型前向传播，计算预测值，输出为概率的 logit
                outputs = model(inputs)
                # 对 logits 执行 softmax 函数，得到预测各个类别的概率，形状为 (N, D) D 为类别数
                probs = nn.Softmax(dim=1)(outputs)
                # 每一行取概率最大的 idx，对应模型输出的预测类别
                # 该函数返回两个对象，第一个对象为最大值，第二个对象为最大值所在的 idx，我们需要第二个对象
                preds = torch.max(probs, 1)[1]
                # 模型输出的损失函数
                loss = criterion(outputs, labels)

                # 损失函数反向传播，求出模型梯度，模型梯度会存储在参数的 grad 属性中，供 optimizer 使用
                loss.backward()
                # 裁剪过大的梯度，防止出现 nan
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                # optimizer 使用求出的梯度，更新模型
                optimizer.step()

                # 记录损失函数
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_cnt += len(labels)

            round_loss = running_loss / total_cnt
            round_acc = running_corrects / total_cnt

            print('CLIENT {} training ok, running loss: {}, running accuracy: {}'.format(
                self.server_idx, round_loss, round_acc))

            # 训练结束，模型放回 CPU
            model.to(torch.device('cpu'))

            # 将更新后的模型发送回 server
            print('CLIENT {} start to send the trained model to server...'.format(self.server_idx))
            send_large_obj(self.cl_conn, model.state_dict(), start_tag=config.OP_TRANSFER_MODEL_START,
                           end_tag=config.OP_TRANSFER_MODEL_END)
            print('CLIENT {} send OK'.format(self.server_idx))

            # 上传指标，便于中间记录
            self.cl_conn.send(
                (config.OP_SEND_METRIC,
                 {'loss': round_loss, 'accuracy': round_acc})
            )
