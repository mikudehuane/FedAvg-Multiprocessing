import torch
import torch.nn as nn


class C3D(nn.Module):
    """3D 卷积模型
    """

    def __init__(self, num_classes):
        """模型初始化

        Args:
            num_classes (int): 类别数
        """
        super(C3D, self).__init__()

        # 以下为各层参数的创建，各层作用可参考 forward 函数
        # 每一层都是一个 nn.Module 的子类，对象内包含了其模型参数
        # 对象可以调用，输入数据输出该层的运算结果

        # Conv3d 是一个 3D 卷积层，参数如下
        # - 3：输入的张量有 3 个 channel（RGB）
        # - 64：创建了 64 个卷积核（参数不同），每个卷积核对输入做重复的操作，输出一个特征图，所有 64 个特征图在 C 维度上重新拼接
        # - kernel_size：每个卷积核的大小，这么大的一个卷积核，在 T, H, W 三个维度上平移，每个位置进行输出
        # - padding：给输入外面填充 0，当 pad * 2 + 1 = kernel 时，输出和输入在 T, H, W 这三个维度上的大小不会有变化
        #     因为 conv 语义上不希望损失信息，所以一般的 conv 都是这么设置的
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # 卷积后传入池化层，输入的一个视频假设为 T, H, W 形状
        # kernel_size 为 t, h, w 形状
        #   那么输入想象成一个长方体，其中取出 t, h, w 的小长方体，每个小长方体内，所有元素取最大值，仅输出该最大值，从而输出的维度降低
        # stride 设置为和 kernel_size 相同，表示这些长方体紧密相邻不相交，设小就是长方体相交，设大就是长方体间有空隙
        #   经典用法就是设为相等
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # 前一层卷积输出了 64 个 channel，因此这层卷积要输入 64 个 channel
        # 输出设置为 128 个 channel，下一次卷积就应该输入 128 个 channel
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        # 全连接层占内存过大，默认情况有全连接层，但是为了显存，删掉了，测试过不会很影响准确率
        # self.fc6 = nn.Linear(8192, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        # 全连接层就是一个矩阵乘法，内部是一个 (8192, num_classes) 的矩阵，与输入进行矩阵乘法，输出一个 num_classes 列的矩阵
        self.fc8 = nn.Linear(8192, num_classes)

        # 会随机删除一些输入，防止过拟合，目前的模型结构中因为删除了全连接，并没有使用
        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU(inplace=True)  # inplace 操作来节省内存

        # 初始化模型参数，kaiming_normal_ 其实就是 pytorch 的默认初始化方法，这一个操作不必要，保险起见加上
        # kaiming_normal_ 初始化会让每一层的输出基本保持在均值为 0，方差为 1，从而更利于优化
        self._init_weight()

    def forward(self, x):
        # 模型输入是 (N, C, T, H, W)
        # - N: batch size
        # - C: number of channels
        # - T: number of time steps（对应多帧图像）
        # - H: image height
        # - W: image width
        # 输出是 (N, num_classes)

        # 卷积层会将卷积核在 T H W 这三个维度上平移，每个位置输出一个值
        # 相比二维卷积，多了 T 这个维度，从而能捕捉视频帧间的联系
        # 卷积后传入 ReLU 层激活，ReLU 层会将输入中的负值置为 0，正值不变
        x = self.relu(self.conv1(x))
        # 卷积后传入池化层，输入的一个视频假设为 T, H, W 形状
        # kernel_size 为 t, h, w 形状
        # 那么输入想象成一个长方体，其中取出 t, h, w 的小长方体，每个小长方体内，所有元素取最大值，仅输出该最大值，从而输出的维度降低
        x = self.pool1(x)
        # 以上 conv -> relu -> pool 为一个经典小模块，接下来就是重复这个模块多次增强模型的效果

        # 同上的一个小模块，模型参数大小不同
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        # conv -> relu -> conv -> relu -> pool 的一个模块
        # 相比前两个模块，在 pool 前多做了一次卷积
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        # conv -> relu -> conv -> relu -> pool 的一个模块
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        # conv -> relu -> conv -> relu -> pool 的一个模块
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        # 现在输出的值仍然是一个 N C T H W 的张量
        # 为了后续输入线性层（一个矩阵乘法），reshape 为 N, C*T*H*W=8192 的 2D 张量
        x = x.view(-1, 8192)

        # 这两层全连接很占显存 ，删除了
        # x = self.relu(self.fc6(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc7(x))
        # x = self.dropout(x)

        # 输入是一个 N, 8192 的矩阵，fc8 中存储了一个 8192, num_classes 的矩阵
        # 两个矩阵相乘，输出各个类别的预测概率对数
        logits = self.fc8(x)

        return logits

    def load_pretrained_weights(self, model_path):
        """加载预训练模型（并非我们代码存储的，而是网上下载的预训练模型，因此需要参数名映射）"""
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
            # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # fc6
            # "classifier.0.weight": "fc6.weight",
            # "classifier.0.bias": "fc6.bias",
            # fc7
            # "classifier.3.weight": "fc7.weight",
            # "classifier.3.bias": "fc7.bias",
        }
        pretrained_dict = torch.load(model_path)

        # self_dict 会包括所有原始的参数
        self_dict = self.state_dict()  # 返回一个字典，键为模型参数名，值为模型参数值
        for name in pretrained_dict:
            if name in corresp_name:  # 预训练模型有更多参数，我们只提取需要的
                # 用预训练模型的参数替换
                self_dict[corresp_name[name]] = pretrained_dict[name]
        # 并非 self_dict 的所有参数都被替换，比如 fc8 的参数就没有在 corresp_name 指定，这为模型的最后一层，使用随机初始化的值

        # 将加载好的参数赋值给模型
        self.load_state_dict(self_dict)

    def _init_weight(self):
        # 初始化模型参数，实际上就是 pytorch 的默认行为，可以不需要调用
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_base_params(self):
        """底层的可训练参数

        这是一个生成器，调用该函数返回一个生成器对象，该对象可以在 for 循环中使用
        for 循环每次迭代，都会执行该函数体（从上次冻结的位置开始）
        直到遇到 yield 语句，yield 的参数会返回给 for 循环指定的对象
        此时函数体执行冻结在 yield 的位置，下次迭代，从此继续执行

        该生成器会依次返回模型除最后一层以外的各组参数
        """
        layers = [self.conv1, self.conv2, self.conv3a, self.conv3b, self.conv4a, self.conv4b,
                  self.conv5a, self.conv5b,
                  # self.fc6,
                  # self.fc7
                  ]
        for layer in layers:
            for param in layer.parameters():
                if param.requires_grad:
                    yield param

    def get_top_params(self):
        """最后一层的参数

        该生成器会依次返回模型最后一层的参数（有 weight 和 bias 两个参数）
        """
        layers = [self.fc8]
        for layer in layers:
            for param in layer.parameters():
                if param.requires_grad:
                    yield param
