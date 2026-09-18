import torch
import os
import typing

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int, 
        out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]):
    """
    checkpoint主要用于记录模型在训练过程中的各种参数，当训练因为某种原因被迫终止时，下次训练可以从checkpoint继续
    Pytorch中无论是nn.Module还是Optimizer的核心参数都存储在state_dict()中，这是一个python字典
    """
    # 保存当前状态
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(checkpoint,out)

def load_checkpoint(
        src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
) -> int :
    # 加载字典
    checkpoint = torch.load(src, map_location='cpu')

    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载优化器参数
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']