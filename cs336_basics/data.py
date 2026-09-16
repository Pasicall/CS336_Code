import torch
import numpy as np
import numpy.typing as npt

def get_batch(
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str
) -> tuple[torch.Tensor,torch.Tensor]:
    """
    npt.NDArray: 一维数组（tokens）
    batch_size: 批大小
    context_length: 上下文长度（一次训练样本中的token数量）
    """

    # 确认最大合法索引
    # 每次训练需要取context_length大小的上下文长度，最后一个可用的起点是len(dataset) - context_length - 1
    n = len(dataset)
    max_idx = n - context_length - 1

    # 生成随机采样点
    ix = torch.randint(0, max_idx + 1, (batch_size,) )

    # 提取输入和输出的坐标
    # x ：dataset[i : i+m]
    # y : dataset[i+1 : i+m+1]
    # 现在cpu上提取数据，然后一次性转化为Tensor
    x_stack = [dataset[i : i + context_length] for i in ix]
    y_stack = [dataset[i+1 : i + context_length + 1] for i in ix]

    # 转换为Pytorch张量并移动到指定设备
    x = torch.from_numpy(np.array(x_stack)).to(device).long()
    y = torch.from_numpy(np.array(y_stack)).to(device).long()

    return x , y

