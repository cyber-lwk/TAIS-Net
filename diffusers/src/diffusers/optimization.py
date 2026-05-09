# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for diffusion models."""

import math
from enum import Enum
from typing import Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .utils import logging


logger = logging.get_logger(__name__)


class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    PIECEWISE_CONSTANT = "piecewise_constant"

'''
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = get_constant_schedule(optimizer)
    使用效果
        学习率在整个训练过程中保持不变，始终为初始设定的值（如这里的0.001）。
        适用于训练过程较为平稳，不需要调整学习率的情况，但可能会导致训练后期优化不足或过早停止。
'''
def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)

'''
    Constant with Warmup Scheduler（带热身的常数学习率调度器）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_warmup_steps = 1000
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    使用效果

        在训练的前num_warmup_steps步中，学习率从0线性增加到初始学习率（0.001）。
        热身阶段后，学习率保持为初始学习率不变。
        适用于模型在训练初期需要逐渐适应数据的情况，避免初始学习率过大导致模型不稳定。
'''
def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

'''
    Piecewise Constant Scheduler（分段常数学习率调度器）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    step_rules = "1:1000,0.1:2000,0.01:3000,0.005"
    scheduler = get_piecewise_constant_schedule(optimizer, step_rules)
    使用效果

        根据step_rules定义的规则，学习率在不同的训练阶段保持不同的常数值。
        例如，在前1000步学习率为1×初始学习率（0.001），接下来的2000步学习率为0.1×初始学习率（0.0001），再接下来的3000步学习率为0.01×初始学习率（0.00001），之后的学习率为0.005×初始学习率（0.000005）。
        适用于根据训练阶段的不同需求，灵活调整学习率的情况。
'''
def get_piecewise_constant_schedule(optimizer: Optimizer, step_rules: str, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        step_rules (`string`):
            The rules for the learning rate. ex: rule_steps="1:10,0.1:20,0.01:30,0.005" it means that the learning rate
            if multiple 1 for the first 10 steps, mutiple 0.1 for the next 20 steps, multiple 0.01 for the next 30
            steps and multiple 0.005 for the other steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    rules_dict = {}
    rule_list = step_rules.split(",")
    for rule_str in rule_list[:-1]:
        value_str, steps_str = rule_str.split(":")
        steps = int(steps_str)
        value = float(value_str)
        rules_dict[steps] = value
    last_lr_multiple = float(rule_list[-1])

    def create_rules_function(rules_dict, last_lr_multiple):
        def rule_func(steps: int) -> float:
            sorted_steps = sorted(rules_dict.keys())
            for i, sorted_step in enumerate(sorted_steps):
                if steps < sorted_step:
                    return rules_dict[sorted_steps[i]]
            return last_lr_multiple

        return rule_func

    rules_func = create_rules_function(rules_dict, last_lr_multiple)

    return LambdaLR(optimizer, rules_func, last_epoch=last_epoch)

'''
    Linear Scheduler with Warmup（带热身的线性衰减学习率调度器）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_warmup_steps = 1000
    num_training_steps = 10000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    使用效果

        在前num_warmup_steps步中，学习率从0线性增加到初始学习率（0.001）。
        热身阶段后，学习率从初始学习率线性衰减到0，直到达到num_training_steps。
        适用于需要在训练初期逐渐提高学习率，然后逐渐降低学习率以精细调整模型的情况。
'''
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

'''
    Cosine Scheduler with Warmup（带热身的余弦衰减学习率调度器）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_warmup_steps = 1000
    num_training_steps = 10000
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    使用效果

        在前num_warmup_steps步中，学习率从0线性增加到初始学习率（0.001）。
        热身阶段后，学习率按照余弦函数的形状从初始学习率衰减到0，直到达到num_training_steps。
        与线性衰减相比，余弦衰减在训练后期的学习率变化更加平滑，有助于模型更好地收敛。
'''
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

'''
    Cosine with Hard Restarts Scheduler with Warmup（带热身和硬重启的余弦衰减学习率调度器）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_warmup_steps = 1000
    num_training_steps = 10000
    num_cycles = 2
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles)
    使用效果

        在前num_warmup_steps步中，学习率从0线性增加到初始学习率（0.001）。
        热身阶段后，学习率按照余弦函数的形状从初始学习率衰减到0，然后在达到0时重新开始下一个周期，总共进行num_cycles个周期。
        适用于需要在训练过程中多次重启学习率以避免陷入局部最优的情况。
'''
def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

'''
    Polynomial Decay Scheduler with Warmup（带热身的多项式衰减学习率调度器）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_warmup_steps = 1000
    num_training_steps = 10000
    lr_end = 1e-7
    power = 2.0
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, lr_end, power)
    使用效果

        在前num_warmup_steps步中，学习率从0线性增加到初始学习率（0.001）。
        热身阶段后，学习率按照多项式函数的形状从初始学习率衰减到lr_end，直到达到num_training_steps。
        通过调整power参数，可以控制学习率衰减的速度和形状，适用于需要灵活调整学习率衰减策略的情况。
'''
def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.PIECEWISE_CONSTANT: get_piecewise_constant_schedule,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    step_rules: Optional[str] = None,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: int = 1,
    power: float = 1.0,
    last_epoch: int = -1,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        step_rules (`str`, *optional*):
            A string representing the step rules to use. This is only used by the `PIECEWISE_CONSTANT` scheduler.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_cycles (`int`, *optional*):
            The number of hard restarts used in `COSINE_WITH_RESTARTS` scheduler.
        power (`float`, *optional*, defaults to 1.0):
            Power factor. See `POLYNOMIAL` scheduler
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, last_epoch=last_epoch)

    if name == SchedulerType.PIECEWISE_CONSTANT:
        return schedule_func(optimizer, step_rules=step_rules, last_epoch=last_epoch)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, last_epoch=last_epoch)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
            last_epoch=last_epoch,
        )

    return schedule_func(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, last_epoch=last_epoch
    )

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    def plot_lr_scheduler(optimizer, scheduler, num_steps):
        lrs = []
        for step in range(num_steps):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
        plt.plot(lrs)
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Scheduler')
        plt.show()

    model = torch.nn.Linear(10, 10)

    # 示例：绘制带热身的线性衰减学习率调度器的曲线
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_warmup_steps = 1000
    num_training_steps = 10000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    plot_lr_scheduler(optimizer, scheduler, num_training_steps)