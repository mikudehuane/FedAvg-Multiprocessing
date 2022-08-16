import os
import os.path as osp
import torch

PROJECT_FD = osp.abspath(osp.join(__file__, '..', '..'))
DATA_UCF_FD = r'E:\backup_E\study_E\ML\Dataset\UCF-101'
LOG_FD = osp.join(PROJECT_FD, 'logs')
RES_FD = osp.join(PROJECT_FD, 'res')  # 资源文件的路径，如预训练模型
FIG_FD = osp.join(PROJECT_FD, 'fig')  # 图片结果的路径
CACHE_FD = osp.join(PROJECT_FD, 'cache')  # 缓存文件的路径，比如绘制的损失函数曲线

os.makedirs(LOG_FD, exist_ok=True)
os.makedirs(CACHE_FD, exist_ok=True)

# 选择了五个类别来做，这样模型最后一层小一点，能塞进显存
SELECTED_CLASSES = {1, 3, 4, 5, 10}
NUM_CLASSES = len(SELECTED_CLASSES)

DEVICE = torch.device('cuda:0')

NUM_CLIENTS = 3  # 客户端数量

OP_GET_ROUND = 'get_round'
OP_GET_NSTEP = 'get_nstep'
# 开始传输模型参数
OP_TRANSFER_MODEL_START = 'transfer_model_start'
# 停止传输模型参数
OP_TRANSFER_MODEL_END = 'transfer_model_end'
OP_SEND_METRIC = 'send_metric'
