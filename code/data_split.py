"""将数据切分成多份（切分元文件），每份对应一个用户"""

import config
import os.path as osp


def main():
    num_splits = 3  # 3 个子进程
    input_fp = osp.join(config.DATA_UCF_FD, 'trainlist01.txt')
    output_fps = [osp.join(config.DATA_UCF_FD, 'trainlist01_{}.txt'.format(i)) for i in range(num_splits)]
    output_fs = [open(output_fp, 'w') for output_fp in output_fps]

    try:
        # 将数据均匀分配到三个子进程
        with open(input_fp, 'r') as f_in:
            for line_idx, line in enumerate(f_in):
                output_fs[line_idx % num_splits].write(line)
    finally:
        [f.close() for f in output_fs]


if __name__ == '__main__':
    main()
