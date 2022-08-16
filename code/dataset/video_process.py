import cv2
import numpy as np


class VideoLoader(object):
    def __init__(
            self,
            train,
            *, resize_height=128, resize_width=171, crop_size=112,
            extract_freq=4, min_frame_count=16
    ):
        """初始化一个视频数据加载器

        Args:
            train: 是否为训练集，训练集可能做数据增强
            resize_height: 输出帧高度
            resize_width: 输出帧宽度
            crop_size: 图片帧大小
            extract_freq: 默认每这么多帧输出一帧
            min_frame_count: 至少输出这么多帧，如果不足会减小 extract_freq，直到 1

        Notes:
            参数列表中 *，表示后面的参数只能通过关键词传入，而不能值传入
        """
        # 预处理超参数
        self.train = train
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.crop_size = crop_size
        self.extract_freq = extract_freq
        self.min_frame_count = min_frame_count

    def __call__(self, video_path):
        frames = self._load_video(video_path)
        frames = frames.astype(np.float32)  # pytorch 需要浮点数输入

        frames = self._crop_video(frames)  # 随机截取视频的一部分（并随机裁剪）

        # 训练集，做 flip 增强
        if self.train and np.random.random() < 0.5:
            self._horizontal_flip_(frames)

        self._normalize_(frames)  # 让均值接近 0

        frames = frames.transpose((3, 0, 1, 2))  # 变成 C, L, H, W
        return frames

    @staticmethod
    def _normalize_(frames):
        for i, frame in enumerate(frames):
            frame -= np.array([[[90.0, 98.0, 102.0]]])

        return frames

    @staticmethod
    def _horizontal_flip_(frames):
        """in-place 水平翻转"""

        for i, frame in enumerate(frames):
            frames[i] = cv2.flip(frames[i], flipCode=1)

    def _crop_video(self, frames):
        # 随机选取时间起点
        time_index = np.random.randint(frames.shape[0] - self.min_frame_count)

        # 随机选取长宽起点
        height_index = np.random.randint(frames.shape[1] - self.crop_size)
        width_index = np.random.randint(frames.shape[2] - self.crop_size)

        frames = frames[
            time_index:time_index + self.min_frame_count,
            height_index: height_index + self.crop_size,
            width_index: width_index + self.crop_size, :
        ]

        return frames

    def _load_video(self, video_path):
        """读取视频文件为 np.array 对象 (L, H, W, C)"""
        # 视频读取对象
        capture = cv2.VideoCapture(video_path)

        # 视频元信息
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 确定帧提取频率
        assert frame_count >= self.min_frame_count, "视频 {} 只有 {} 帧，过短，不足 {} 帧".format(
            video_path, frame_count, self.min_frame_count)
        extract_freq = self.extract_freq
        while frame_count // extract_freq <= self.min_frame_count:
            extract_freq -= 1

        frame_idx = 0  # 帧率 index
        retaining = True  # 是否还有剩余的帧
        frames = []  # 读出的 frame
        while frame_idx < frame_count and retaining:
            retaining, frame = capture.read()
            if frame is None:  # 没有读到帧
                continue

            if frame_idx % extract_freq == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                    frames.append(frame)
            frame_idx += 1
        assert len(frames) >= self.min_frame_count, "代码逻辑有误，视频 {} 只读出 {} 帧，过短，不足 {} 帧".format(
            video_path, len(frames), self.min_frame_count)
        frames = np.stack(frames)

        # Release the VideoCapture once it is no longer needed
        capture.release()
        return frames
