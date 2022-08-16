"""参数服务器进程，训练程序入口，会拉起 client 进程"""
import json
import os
import pickle
import random
import time
from collections import OrderedDict
from torchsummary import summary

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
import os.path as osp

import network
from client import Client
from comm import send_large_obj, recv_large_obj, get_msg
from dataset import UCF101Dataset, VideoLoader
import multiprocessing as mp
import numpy as np


def eval_loader(model, loader, criterion=nn.CrossEntropyLoss()):
    """评估模型，返回准确率和 loss"""
    model.eval()
    with torch.no_grad():  # 在该语句下的子块内执行的所有代码，torch 都不会记录梯度
        # 模型评估结果缓存在这些值里
        cnt_correct = 0
        cnt_total = 0
        loss_sum = 0.0
        # tqdm 提供可视化的进度条，这里将数据集的数据通过 loader 依次取出
        for data in tqdm(loader):
            # 每条数据包括一个输入视频（inputs），和一个 groudtruth 标签（labels）
            inputs, labels = data
            # 放到对应的设备上
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            # 模型运算
            outputs = model(inputs)
            # 求损失函数
            loss = criterion(outputs, labels)
            # 记录损失函数和准确率
            loss_sum += loss.item()
            # 每一行取概率最大的 idx，对应模型输出的预测类别
            # 该函数返回两个对象，第一个对象为最大值，第二个对象为最大值所在的 idx，我们需要第二个对象
            _, predicted = torch.max(outputs.data, 1)
            cnt_total += labels.size(0)
            cnt_correct += (predicted == labels).sum().item()
    return cnt_correct / cnt_total, loss_sum / cnt_total


def main():
    # 超参数设定
    num_local_steps = 5  # 本地训练，每轮迭代次数，每这么多次迭代上传一次模型
    batch_size = 1  # 每次迭代，喂多少条数据，为了降低显存占用，设为了 1
    batch_size_test = 20  # 测试时的 batch size
    lr = 1e-3  # 初始学习率
    num_rounds = 100  # 训练的通讯轮数
    test_every = 1  # 每多少轮迭代测试一次

    # 随机种子指定
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # 日志记录的路径
    log_fd = osp.join(config.LOG_FD, time.strftime("%Y%m%d-%H%M%S"))
    model_fd = osp.join(log_fd, 'models')  # 保存模型的路径，各个模型的评估结果会存储在该目录下同名 json 文件中
    tensorboard_fd = log_fd  # tensorboard 输出的存储路径，存放在根目录便于查看
    # 创建这些文件夹
    os.makedirs(model_fd, exist_ok=True)
    os.makedirs(tensorboard_fd, exist_ok=True)

    clients = dict()  # 存储 client 进程对象
    for client_idx in range(config.NUM_CLIENTS):
        # 创建一个多进程通讯的 Pipe 对象，返回通讯管道的两端，cl_conn 客户进程使用，ps_conn 主进程使用
        cl_conn, ps_conn = mp.Pipe(True)
        # 创建 client 对象，对应一个进程，此时指定一些参数，实际还没有开启进程
        clients[client_idx] = Client(
            client_idx=client_idx, cl_conn=cl_conn, ps_conn=ps_conn,
            trainset_index=osp.join(config.DATA_UCF_FD, 'trainlist01_{}.txt'.format(client_idx)),
            batch_size=batch_size, lr=lr, tensorboard_fd=tensorboard_fd
        )
    # 创建模型
    model = network.C3D(num_classes=config.NUM_CLASSES)
    model.to(device=config.DEVICE)
    summary(model, (3, 16, 112, 112))
    # 加载预训练模型
    model.load_pretrained_weights(osp.join(config.RES_FD, 'ucf101-caffe.pth'))
    model.to(torch.device('cpu'))  # 为了节省显存，主进程上的参数放在 CPU

    # 创建数据集对象，server 只存测试集，用于评估模型
    testset = UCF101Dataset(video_index=osp.join(config.DATA_UCF_FD, 'testlist01.txt'),
                            video_processor=VideoLoader(train=False),
                            selected_classes=config.SELECTED_CLASSES)
    # 数据集对象只负责解析数据，并一条条的返回，但是模型需要一个 batch 一个 batch 的数据（一次多条数据）
    # DataLoader（pytorch 提供的）就负责拼接 batch
    # shuffle=False 表示数据集按照原有顺序返回
    test_loader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)

    # 训练初始化（损失函数等）
    test_criterion = nn.CrossEntropyLoss().to(config.DEVICE)

    # 日志记录对象，该对象写下的内容可以通过 tensorboard 查看，在项目根目录下执行以下命令即可
    # >>> tensorboard --logdir=logs
    writer = SummaryWriter(tensorboard_fd)

    # 开启客户进程，这时客户进程才真正启动
    for client_idx, client in clients.items():
        client.start()

    for round_idx in range(num_rounds):
        if round_idx % test_every == 0:
            # 评估上一轮的模型
            model.to(config.DEVICE)
            # 评估模型
            test_acc, test_loss = eval_loader(model, test_loader, criterion=test_criterion)
            # 向 tensorboard 写入统计信息，从而可以在网页上看到
            writer.add_scalar('test_loss_round', test_loss, round_idx)
            writer.add_scalar('test_acc_round', test_acc, round_idx)
            print("\n{}/{} round start, testset Loss: {} Acc: {}".format(
                round_idx, num_rounds, test_loss, test_acc))
            # 放回 CPU，释放显存
            model.to(torch.device('cpu'))

            # 保存模型和测试数据
            model_name = 'C3D_{}'.format(round_idx)
            # 保存模型
            torch.save(model.state_dict(), osp.join(model_fd, '{}.pth'.format(model_name)))
            # 保存测试结果
            json.dump({'test_loss': test_loss, 'test_accuracy': test_acc},
                      open(osp.join(model_fd, '{}.json'.format(model_name)), 'w'))

        for client_idx, client in clients.items():
            # 发送一些配置信息，通知 client 开始下一轮训练
            client.ps_conn.send((config.OP_GET_ROUND, round_idx))  # 发送当前 round 号
            client.ps_conn.send((config.OP_GET_NSTEP, num_local_steps))  # 发送此 round 应迭代的次数
            print('SERVER start to send client {} the current model...'.format(client_idx))
            # 发送模型到 client 进程
            send_large_obj(client.ps_conn, model.state_dict(), start_tag=config.OP_TRANSFER_MODEL_START,
                           end_tag=config.OP_TRANSFER_MODEL_END)
            print('SERVER send ok'.format(client_idx))

        # 准备聚合模型参数，为了求均值，先初始化一个参数均为 0 的字典
        state_dict = OrderedDict({key: torch.zeros_like(val) for key, val in model.state_dict().items()})
        for client_idx, client in clients.items():
            # 等待并接受 client 训练好的模型
            print('SERVER querying the trained model from client {}...'.format(client_idx))
            cl_state = recv_large_obj(
                client.ps_conn, start_tag=config.OP_TRANSFER_MODEL_START,
                end_tag=config.OP_TRANSFER_MODEL_END)  # 发送模型到 client 进程
            print('SERVER received model from client {}'.format(client_idx))

            print('SERVER aggregating the model')
            # 从终端发送来的模型参数，累积求和，FedAvg 即求各个终端更新后模型的均值
            # 数学上等价于用各个终端模型的更新量，的均值，更新全局模型
            for key, val in cl_state.items():
                state_dict[key] += val

            # 从 client 接受这一轮训练的 metric 值，记录
            round_metrics, = get_msg(client.ps_conn, config.OP_SEND_METRIC)
            round_loss = round_metrics['loss']
            round_acc = round_metrics['accuracy']

            writer.add_scalar('train_loss_round_{}'.format(client_idx), round_loss, round_idx)
            writer.add_scalar('train_acc_round_{}'.format(client_idx), round_acc, round_idx)

        # 求均值
        for key, val in state_dict.items():
            state_dict[key] = val / len(clients)
        # 更新模型参数
        model.load_state_dict(state_dict)

    for client_idx, client in clients.items():
        # 发送信息请求 client 关闭，client 中实现了逻辑，round_idx 为 -1 时，关闭
        client.ps_conn.send((config.OP_GET_ROUND, -1))


if __name__ == '__main__':
    main()
