import torch
import math
from torch.optim import Optimizer
from collections.abc import Iterable

class AdamW(Optimizer):
    # 优化器的目标是找到一组参数theta使得模型训练的损失最小
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
        # 基本参数检查
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        # 将超参数存入 defaults 字典
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params,defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.no_grad():
                loss = closure()

        # 拿参数
        for group in self.param_groups:
            beta1,beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']
            """
            Optimizer
                    --states[p]
                                ---
                                    step
                                    m
                                    v
            """    
            # 获取当前梯度的参数p
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                # 建立历史状态
                state = self.state[p]

                # 状态初始化（第一运行步）
                if len(state) == 0:
                    state['step'] = 0
                    # m: 一阶矩
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v: 二阶矩
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # 更新矩估计
                #  m = beta_1*m_t-1 + (1-beta_1)*g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                #  v = beta_2*v_t-1 +(1-beta_2)*g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 计算偏差校正后的参数，为了消除m，v初始值为0时带来的误差
                bia_correction1 = 1 - beta1 ** t
                bia_correction2 = 1 - beta2 ** t
                step_size = lr * (math.sqrt(bia_correction2) / bia_correction1)
                # 更新参数 theta = theta - alpha_t * m / (sqrt(v) + eps) = theta - alpha_t * m / (sqrt(v) + eps)
                denom = exp_avg_sq.sqrt().add_(eps)
                # p.addcdiv(tensor1,tensor2, value) --> p = p + value * (tensor1 / tensor2)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # 解耦权重衰减（AdawmW的特性）
                #  theta = p - lr*lambda*theta
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)
        return loss






            
            

