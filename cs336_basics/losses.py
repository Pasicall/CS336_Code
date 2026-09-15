import torch

def cross_entropy(logits:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
    """
    logits(o_y): 形状[batch_size,seq,vocab_size] 的预测分值
    targets: 形状为[batch_size, seq] 的真实id
    loss = M + log sum(exp(o_y - M)) - o_y 其中M为logits的最大值
    """
    
    # 1.计算logits的最大值M
    m = torch.max(logits, dim=-1, keepdim=True).values

    # 2.提取目标位置的logits(o_y)
    # gather需要使用unsqueeze将targets进行维度扩展后再使用
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 3.计算 Log-Sum-Exp项
    shifted_logits = logits - m
    log_sum_exp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logits),dim=-1))

    # 4.计算单个token的损失
    loss = log_sum_exp - target_logits

    return torch.mean(loss)
