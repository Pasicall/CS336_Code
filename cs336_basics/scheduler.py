import math

def get_lr_cosine_scheduler(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int
):
    """
    it: 当前迭代步数(t)
    max_learning_rate: 最大学习率(alpha_max)
    min_learning_rate: 最小学习率(alpha_min)
    warmup_iters: 预热步数(T_w)
    cosine_cycle_iters: 退火步数(T_c)
    """

    # Warm-up阶段，从小步长开始逐渐增加
    # alpha = alpha_max * it / T_w
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    # Post-Annealing阶段，保持最低速滑行
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    # Cosine Annealing阶段，利用cos函数平稳下落
    # 该阶段的局部时间应该处理为 ： it - warmup_iters
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) 
    # 计算余弦系数
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))

    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
