import json
import os
import re
import sys
import time
from uuid import uuid4

import cv2
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QGridLayout, QFileDialog, \
    QMessageBox

from PyQt5 import QtCore, QtGui, QtWidgets, Qt
import os.path as osp

from matplotlib import pyplot as plt
from torch import nn

import config
import network
from dataset import UCF101Dataset, VideoLoader


class App(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # 默认字体
        default_font = QtGui.QFont()
        default_font.setPointSize(15)

        # 三个按钮
        model_btn = QPushButton("选择训练模型")
        model_btn.clicked.connect(self.choose_model)
        model_btn.setFont(default_font)
        video_btn = QPushButton("上传视频文件")
        video_btn.clicked.connect(self.upload_video)
        video_btn.setFont(default_font)
        ok_btn = QPushButton("开始识别")
        ok_btn.clicked.connect(self.predict)
        ok_btn.setFont(default_font)
        # 三个按钮放在一排
        hbox = QHBoxLayout()
        hbox.addStretch(1)  # 左边空出来，右对齐
        hbox.addWidget(model_btn)
        hbox.addWidget(video_btn)
        hbox.addWidget(ok_btn)

        # 视频展示垂直布局
        vbox = QGridLayout()
        title_label = QtWidgets.QLabel("UCF101 视频识别测试")
        # 设置字体
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setItalic(True)
        font.setWeight(75)
        title_label.setFont(font)

        vbox.addWidget(title_label, 1, 0, 1, 3)
        # 展示视频的位置
        self.video_frame = QtWidgets.QLabel()
        # 显示白幕
        self.set_pixmap(self.video_frame, osp.join(config.FIG_FD, 'white.png'))
        vbox.addWidget(self.video_frame, 3, 0, 1, 1)
        # 显示准确率曲线
        self.acc_frame = QtWidgets.QLabel()
        self.set_pixmap(self.acc_frame, osp.join(config.FIG_FD, 'white.png'))
        vbox.addWidget(self.acc_frame, 3, 1, 1, 1)
        # 显示损失函数曲线
        self.loss_frame = QtWidgets.QLabel()
        self.set_pixmap(self.loss_frame, osp.join(config.FIG_FD, 'white.png'))
        vbox.addWidget(self.loss_frame, 3, 2, 1, 1)
        self.model_label = QtWidgets.QLabel("模型文件：")
        self.model_label.setFont(default_font)
        vbox.addWidget(self.model_label, 4, 0, 1, 2)
        # 按钮视图的行
        vbox.addLayout(hbox, 5, 0, 1, 2)

        self.setLayout(vbox)

        # 主窗口 UI
        self.setGeometry(800, 600, 800, 600)
        self.setWindowTitle('联合学习系统')

        self.state = dict()  # 记录当前的状态，比如选中的视频

        self.model = network.C3D(num_classes=config.NUM_CLASSES)  # 创建未训练的模型
        self.model.eval()
        self.model.to(config.DEVICE)  # 将模型放到指定的设备上

        # 获取类别 idx 到类别名的映射关系
        class_index = osp.join(config.DATA_UCF_FD, 'classInd.txt')
        # 类别原始 idx -> 类别名
        self.class_index = dict()
        with open(class_index, 'r') as f:
            for line in f:
                class_idx, class_name = line.strip().split()
                self.class_index[int(class_idx)] = class_name
        # 类别在模型中 idx -> 类别原始 idx
        self.idx_mapping = {
            idx_key: idx_val
            # selected_classes 排序后填充字典，从而避免不同时候调用生成的字典不一致
            for idx_key, idx_val in enumerate(sorted(list(config.SELECTED_CLASSES)))
        }
        debug = 1

    def predict(self):
        if 'video_path' not in self.state:
            QMessageBox.critical(self, '错误', '请先上传视频')
            return
        if 'model_path' not in self.state:
            QMessageBox.critical(self, '错误', '请先上传模型')
            return

        # 读取视频文件
        frames = VideoLoader(train=False)(video_path=self.state['video_path'])
        # 将视频帧转换成 tensor
        frames = torch.from_numpy(frames).to(config.DEVICE)
        # 多出一个 batch 维度
        frames = torch.unsqueeze(frames, 0)
        with torch.no_grad():
            logits = self.model(frames)  # 模型前向传播，计算预测值，输出为概率的 logit
            # 对 logits 执行 softmax 函数，得到预测各个类别的概率，形状为 (N, D) D 为类别数
            probs = nn.Softmax(dim=1)(logits)
            # 每一行取概率最大的 idx，对应模型输出的预测类别
            preds = torch.max(probs, 1)[1]

            # 长为 1 的 batch
            pred = preds[0].item()
            prob = probs[0][pred].item()
            # 预测的类别名
            pred = self.class_index[self.idx_mapping[pred]]
            QMessageBox.information(self, '预测结果', "预测动作：{}，概率：{}".format(pred, prob))

    def choose_model(self):
        """选择训练模型"""
        model_fd = QFileDialog.getExistingDirectory(self, "选择模型所在的文件夹", config.LOG_FD)
        if model_fd:
            # 校验并读取文件夹的内容
            file_names = os.listdir(model_fd)  # 文件夹下的所有文件
            # 正则匹配文件名
            model_name_pat = re.compile(r'(C3D_(\d+))\.pth')
            # 匹配出这个文件夹下，模型对应了哪些迭代
            round_idxs = []
            for file_name in file_names:
                # re.match 会返回一个 match 的结果，
                matched = re.match(model_name_pat, file_name)
                # 匹配到了
                if matched is not None:
                    # group(2) 会从匹配结果中，寻找第二个括号内的内容，即 (\d+)，\d 表示任意数字，+ 表示至少出现一次，可以多次
                    round_idxs.append(int(matched.group(2)))

            # 根据 round_idx 排序
            round_idxs.sort()

            # 如果没找到模型，报错
            if not round_idxs:
                QMessageBox.critical(self, '错误', '选择的文件夹中没有存储模型')
                return

            # 依次读取评估结果
            losses = []
            accs = []
            for round_idx in round_idxs:
                # 存储结果的文件名
                meta_name = 'C3D_{}.json'.format(round_idx)
                meta_path = osp.join(model_fd, meta_name)
                # 没找到统计信息，报错
                if not osp.exists(meta_path):
                    QMessageBox.critical(self, '错误', '没有找到需要的模型评估结果文件 {}'.format(meta_path))
                    return
                metrics = json.load(open(meta_path))
                losses.append(metrics['test_loss'])
                accs.append(metrics['test_accuracy'])

            # 绘制损失函数曲线
            plt.plot(round_idxs, losses)
            plt.title('Test Loss')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            # 创建唯一 ID 的图片，并存下
            fig_path = osp.join(config.CACHE_FD, '{}.png'.format(uuid4()))
            plt.savefig(fig_path)
            plt.close()
            self.set_pixmap(self.loss_frame, fig_path)

            # 绘制准确率曲线
            plt.plot(round_idxs, accs)
            plt.title('Test Accuracy')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            # 创建唯一 ID 的图片，并存下
            fig_path = osp.join(config.CACHE_FD, '{}.png'.format(uuid4()))
            plt.savefig(fig_path)
            plt.close()
            self.set_pixmap(self.acc_frame, fig_path)

            # 取最后一轮的模型
            model_path = osp.join(model_fd, 'C3D_{}.pth'.format(round_idxs[-1]))

            self.state['model_path'] = model_path
            model_name = osp.basename(model_path)
            self.model_label.setText("模型文件：" + model_name)
            self.model.load_state_dict(torch.load(model_path))

    def set_pixmap(self, label_frame, fig_path):
        """将指定路径的图片显示在 label_frame 上"""
        image = cv2.imread(fig_path)
        # resize 到窗口的大小
        image = cv2.resize(image, (self.video_frame.size().width(), self.video_frame.size().height()))
        # 颜色转换
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        label_frame.setPixmap(QPixmap(image))

    def upload_video(self):
        """上传视频并展示在 self.video_frame"""

        video_path, ok = QFileDialog.getOpenFileName(self, "上传视频", config.DATA_UCF_FD, "Video Files (*.avi *.mp4)")
        if ok:
            self.state['video_path'] = video_path

            # 以下代码读取一帧视频展示
            video_cap = cv2.VideoCapture(video_path)
            # 创建视频读取对象
            ret, image = video_cap.read()
            if ret:
                # resize 到窗口的大小
                image = cv2.resize(image, (self.video_frame.size().width(), self.video_frame.size().height()))
                # 颜色转换
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                video_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                self.video_frame.setPixmap(QPixmap(video_img))
                # 创建一个数据集，其中只有待预测的视频，避免重写图片预处理代码


class LoginWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setObjectName("login_MainWindow")
        self.setEnabled(True)
        self.resize(575, 392)
        self.setAnimated(True)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(90, 30, 600, 200))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setItalic(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(100, 200, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 150, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(180, 200, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.lineEdit.setFont(font)
        self.lineEdit.setText("")
        self.lineEdit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(180, 150, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(180, 280, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 575, 22))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)

        self.retranslate_ui()
        QtCore.QMetaObject.connectSlotsByName(self)

        self.init()

        self.ui_main = None

    def init(self):
        self.pushButton.clicked.connect(self.login_button)  # 连接槽

    def login_button(self):
        if self.lineEdit.text() == "":
            QMessageBox.warning(self, '警告', '密码不能为空，请输入！')
            return None

        # 直接展示主界面
        self.ui_main = App()
        self.ui_main.show()
        self.close()

    def retranslate_ui(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("login_MainWindow", "登录"))
        self.label_3.setText(_translate("login_MainWindow", "欢迎使用联合学习系统"))
        self.label_2.setText(_translate("login_MainWindow", "密码："))
        self.label.setText(_translate("login_MainWindow", "用户："))
        self.lineEdit_2.setText(_translate("login_MainWindow", "admin"))
        self.pushButton.setText(_translate("login_MainWindow", "登录"))


def main():
    app = QApplication(sys.argv)
    ex = LoginWindow()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
