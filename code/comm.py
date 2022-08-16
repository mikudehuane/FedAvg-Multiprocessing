"""进程通讯相关函数"""

import pickle
import config


def send_large_obj(conn, obj, *, chunk_size=2000000, start_tag, end_tag):
    """通过管道发送大的对象，因为管道有大小限制，所以要先序列化，然后分批发送

    Args:
        conn: 管道的发送端
        obj: 待发送的对象
        chunk_size: 每次发送的数据大小，默认为 2M
        start_tag: 开始信号
        end_tag: 结束信号
    """
    # 序列化传入的对象
    obj_s = pickle.dumps(obj)

    conn.send((start_tag,))  # 发送开始信号
    start_idx = 0
    while len(obj_s) > start_idx:
        conn.send(obj_s[start_idx: start_idx + chunk_size])
        start_idx += chunk_size  # 可能会超过 len(obj_s)，但是同样也会让循环终止
    # 发送停止信号
    conn.send((end_tag,))


def get_msg(conn, tag):
    """等待管道另一端发送对应 tag 的信息，不符合的信息会被舍弃

    Args:
        conn: 通讯 Pipe 的接收端
        tag: 希望接受的信息
            原始传入的信息会是一个 tuple，tuple 的第一个元素为 OP 的名称，应为 config.OP_* 常量
            tag 会与该 OP 名匹配

    Returns:
        接收到的信息内容，即 msg[1:]
    """
    while True:
        msg = conn.recv()  # 等待 server 发送的信息，该 API 会在有信息时立刻接受
        if msg[0] == tag:  # 只收与指定 tag 相符的信息，丢弃其他的
            return msg[1:]


def recv_large_obj(conn, *, start_tag, end_tag):
    """从 server 接受一个较大的对象，比如模型参数

    Args:
        conn: 通讯 Pipe 的接收端
        start_tag: 开始信号
        end_tag: 结束信号
    """
    get_msg(conn, start_tag)  # 等待 server 发送开始信号
    bytes_buffer = []
    while True:
        msg = conn.recv()
        if msg[0] == end_tag:  # 收到终止信号，结束
            break
        else:
            bytes_buffer.append(msg)  # 收到数据，添加到缓存
    return pickle.loads(b''.join(bytes_buffer))  # 将缓存中的数据转换为对象
