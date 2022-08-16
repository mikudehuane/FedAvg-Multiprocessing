# 说明
使用的分类模型为 C3D（3D convolution）。
## 代码框架
代码框架如下，[app.py](code/app.py) 为预测 GUI 程序的入口脚本，[server.py](code/server.py) 为联合学习训练的入口脚本，[dataset](code/dataset) 模块为数据集处理相关代码，[client.py](code/client.py) 为客户端进程逻辑，[comm.py](code/comm.py) 实现了进程间通讯，[config.py](code/config.py) 为一些路径等配置，[data_split.py](code/data_split.py) 脚本用于切分三个 client 的数据集，[network.py](code/network.py) 实现了 C3D 视频分类神经网络

![代码结构](fig/code.png)

[res](res) 目录下存储了预训练模型，[logs](logs) 文件夹下存储运行日志（包括保存的模型）

## 运行前配置
打开 [config.py](code/config.py) 文件，修改 DATA_UCF_FD 为完整数据集存储的目录，目录结构如下（注意红框中的几个文件，为数据集自带的元数据，trainlist01_*.txt 制定了数据集的划分，由 [data_split.py](code/data_split.py) 生成。

![数据集文件夹](fig/data.png)

接下来指定需要考虑的类别，默认只选择了五个类别，如有需要增加类别，直接修改该参数为对应类别的 idx 组成的 set 即可（不需要改动其他东西）。

预训练模型请从 https://pan.baidu.com/s/1saNqGBkzZHwZpG-A5RDLVw 下载到 [res](res) 文件夹。

## 运行环境
代码在 pycharm 下编辑运行，python 版本 3.7.9，python 完整的环境和包安装情况如下（不一定全部需要）。
pytorch 注意安装 GPU 版本，在 https://pytorch.org/get-started/locally/ 可以根据电脑信息查找到对应命令（建议 pip 安装）。
安装后在 python 内用 torch.cuda.is_available() 检查 GPU 是否可用。
```
absl-py                   1.0.0
altgraph                  0.17.2
anyio                     3.5.0
argon2-cffi               21.3.0
argon2-cffi-bindings      21.2.0
attrs                     21.4.0
av                        9.2.0
Babel                     2.10.1
backcall                  0.2.0
beautifulsoup4            4.11.1
bleach                    5.0.0
cachetools                5.0.0
certifi                   2021.10.8
cffi                      1.15.0
charset-normalizer        2.0.12
colorama                  0.4.4
cycler                    0.11.0
debugpy                   1.6.0
decorator                 5.1.1
defusedxml                0.7.1
entrypoints               0.4
fastjsonschema            2.15.3
fonttools                 4.28.5
future                    0.18.2
google-auth               2.6.3
google-auth-oauthlib      0.4.6
grpcio                    1.44.0
idna                      3.3
importlib-metadata        4.11.3
importlib-resources       5.7.1
ipykernel                 6.13.0
ipython                   7.33.0
ipython-genutils          0.2.0
jedi                      0.18.1
Jinja2                    3.1.2
joblib                    1.1.0
json5                     0.9.6
jsonschema                4.4.0
jupyter-client            7.3.0
jupyter-core              4.10.0
jupyter-server            1.17.0
jupyterlab                3.3.4
jupyterlab-pygments       0.2.2
jupyterlab-server         2.13.0
kiwisolver                1.3.2
Markdown                  3.3.6
MarkupSafe                2.1.1
matplotlib                3.5.1
matplotlib-inline         0.1.3
mistune                   0.8.4
nbclassic                 0.3.7
nbclient                  0.6.0
nbconvert                 6.5.0
nbformat                  5.3.0
nest-asyncio              1.5.5
notebook                  6.4.11
notebook-shim             0.1.0
numpy                     1.21.5
oauthlib                  3.2.0
opencv-python             4.5.5.64
packaging                 21.3
pandocfilters             1.5.0
parso                     0.8.3
pefile                    2021.9.3
pickleshare               0.7.5
Pillow                    8.4.0
pip                       20.1.1
prometheus-client         0.14.1
prompt-toolkit            3.0.29
protobuf                  3.20.0
psutil                    5.9.0
pyasn1                    0.4.8
pyasn1-modules            0.2.8
pycparser                 2.21
Pygments                  2.12.0
pyinstaller               5.0.1
pyinstaller-hooks-contrib 2022.4
pyparsing                 3.0.6
PyQt5                     5.15.6
PyQt5-Qt5                 5.15.2
PyQt5-sip                 12.10.1
PyQt5-stubs               5.15.6.0
pyrsistent                0.18.1
python-dateutil           2.8.2
pytz                      2022.1
pywin32                   303
pywin32-ctypes            0.2.0
pywinpty                  2.0.5
pyzmq                     22.3.0
requests                  2.27.1
requests-oauthlib         1.3.1
rsa                       4.8
scikit-learn              1.0.2
scipy                     1.7.3
Send2Trash                1.8.0
setuptools                47.1.0
six                       1.16.0
sniffio                   1.2.0
soupsieve                 2.3.2.post1
tensorboard               2.8.0
tensorboard-data-server   0.6.1
tensorboard-plugin-wit    1.8.1
terminado                 0.13.3
threadpoolctl             3.1.0
tinycss2                  1.1.1
torch                     1.11.0+cu113
torchaudio                0.11.0+cu113
torchvision               0.12.0+cu113
tornado                   6.1
tqdm                      4.64.0
traitlets                 5.1.1
typing-extensions         4.1.1
urllib3                   1.26.9
wcwidth                   0.2.5
webencodings              0.5.1
websocket-client          1.3.2
Werkzeug                  2.1.1
wheel                     0.37.1
wxPython                  4.1.1
zipp                      3.8.0
```
