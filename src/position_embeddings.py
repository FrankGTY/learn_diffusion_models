import math
import torch
from torch import nn


class SinusidalPositionEmbeddings(nn.Module):
    """
    上面的代码是一个 Python 实现的正弦位置嵌入类，用于在神经网络中添加正弦位置嵌入。正弦位置嵌入是一种常用的技术，用于为序列数据中的每个位置添加一个唯一的嵌入向量。这些嵌入向量可以帮助神经网络更好地理解序列数据中的位置信息。

    SinusidalPositionEmbeddings 类是 nn.Module 类的子类，它是 PyTorch 中所有神经网络模块的基类。该类接受一个参数 dim，它指定嵌入向量的维度。

    forward 方法接受一个参数 time，它是一个张量，表示序列中每个位置的时间步。在 forward 方法中，首先获取 time 张量的设备类型，然后计算嵌入向量的一半维度。接下来，计算正弦位置嵌入的值，这是通过将 10000 取对数，然后除以嵌入向量的一半维度减去 1 来计算的。然后使用张量指数函数计算嵌入向量的值，并将其与时间张量相乘。最后，将正弦和余弦值连接在一起，并返回嵌入向量。

    总的来说，SinusidalPositionEmbeddings 类提供了一种方便的方法来为序列数据中的每个位置添加正弦位置嵌入。这种技术可以帮助神经网络更好地理解序列数据中的位置信息，从而提高模型的性能。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    