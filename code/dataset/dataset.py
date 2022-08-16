import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import config


class UCF101Dataset(Dataset):
    """UCF101 数据集

    关于视频预处理：
    1. 视频每一帧 resize 到 (self.resize_height, self.resize_width)
    2. 每 self.extract_freq 提取一帧，并保证最终得到至少 self.min_frame_count 帧（若不足则减小 extract_freq）
    3. 对提取出的帧，随机取出连续的 self.min_frame_count 帧，并且随机选取固定的 (self.crop_size, self.crop_size) 区域裁剪
    4. 对于训练集，以 0.5 的概率左右翻转
    5. 对每一帧进行归一化处理（均值归为 0，暂不处理方差），并且将其转换为 float32 格式
    6. 转置到 C, L, H, W 的格式

    Attributes:
        class_index: 类别索引（类别名全部取小写字母）
        class_index_reverse: class_idx -> class_name
        video_index: 文件索引，该数据集负责的所有文件的绝对路径
        idx_mapping (dict): 选择的类别，键为类别的原始 idx，值为从 0 开始连续的 idx
            因此注意，实际输出的 label 经过了二级映射，即 label = idx_mapping[class_index[class_name]]
        labels: 文件的 label，从文件名中提取
    """

    def __init__(
            self, video_index, video_processor, selected_classes=None, *,
            class_index=osp.join(config.DATA_UCF_FD, 'classInd.txt'),
    ):
        """初始化数据集

        Args:
            video_index: 存放视频路径的列表，或传入存储该信息文件的路径，会读取为所需的列表（__index__ 读取文件时的 offset 参照该列表）
            video_processor: 处理视频数据的对象，输入一个视频路径，读取出需要的内容
            selected_classes: 传入一个 set，指定哪些类别是关注的，其他类别都会被忽略，默认选择全部的类别
            class_index: 视频类别名 -> 视频类别编号的字典，或传入存储该信息文件的路径，会读取为所需的字典（视频所在文件夹名为类别名）
                视频文件格式：一行一个相对 config.DATA_UCF_FD 的路径（可以有空白字符分割的两列，第二列表类别，第二列会被校验与文件名一致）
        """
        self.video_processor = video_processor

        # 处理 class_index
        if isinstance(class_index, str):
            self.class_index = dict()
            with open(class_index, 'r') as f:
                for line in f:
                    class_idx, class_name = line.strip().split()
                    self.class_index[class_name.lower()] = int(class_idx)
        else:
            self.class_index = class_index
        self.class_index_reverse = dict()
        for class_name, class_idx in self.class_index.items():
            self.class_index_reverse[class_idx] = class_name
        assert len(self.class_index) == 101, 'class_index 应该包含 101 个类别'
        # 处理 selected_classes
        if selected_classes is None:
            selected_classes = set(self.class_index.values())  # 选取全部的 idx
        # 生成 idx_mapping
        self.idx_mapping = {
            idx_key: idx_val
            # selected_classes 排序后填充字典，从而避免不同时候调用生成的字典不一致
            for idx_val, idx_key in enumerate(sorted(list(selected_classes)))
        }
        # 处理 video_index，生成索引与 label
        if isinstance(video_index, str):
            self_video_index = []
            with open(video_index, 'r') as f:
                for line in f:
                    line_s = line.strip().split()
                    if len(line_s) == 1:
                        video_path = osp.join(config.DATA_UCF_FD, line_s[0])
                    elif len(line_s) == 2:
                        f_rpath, class_idx = line_s
                        cls_from_path = self._get_class_from_path(f_rpath)
                        cls_from_file = self.class_index_reverse[int(class_idx)]
                        assert cls_from_file == cls_from_path, \
                            "从行 {} 读取出不一致的类别 {} 与 {}".format(line, cls_from_path, cls_from_file)
                        video_path = osp.join(config.DATA_UCF_FD, f_rpath)
                    else:
                        raise RuntimeError("行 {} 不符合格式".format(line))
                    self_video_index.append(video_path)
        else:
            self_video_index = video_index
        # 生成标签数据
        self_labels = [self.class_index[self._get_class_from_path(video_path)] for video_path in self_video_index]

        # 过滤标签数据
        self.video_index = []
        self.labels = []
        for idx, label in enumerate(self_labels):
            if label in self.idx_mapping:
                self.video_index.append(self_video_index[idx])
                self.labels.append(self.idx_mapping[label])

    def __getitem__(self, item):
        # 加载元信息
        video_path = self.video_index[item]
        label = self.labels[item]

        frames = self.video_processor(video_path)

        return frames, label

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _get_class_from_path(video_path):
        return osp.basename(video_path).split('_')[1].lower()
