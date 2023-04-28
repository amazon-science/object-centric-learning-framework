import pytest
from torch.optim import Optimizer

from ocl import scheduling


class MockOptimizer(Optimizer):
    def __init__(self, lr):
        self.param_groups = [{"lr": lr}]
        self.last_lr = None

    def step(self):
        self.last_lr = self.param_groups[0]["lr"]


@pytest.mark.parametrize(
    "total_steps,warmup_steps,min_lr", [(400, 100, 0.0), (400, 0, 0.0), (400, 100, 0.0001)]
)
@pytest.mark.parametrize("expect_error", [False, True])
def test_cosine_annealing_with_warmup(total_steps, warmup_steps, min_lr, expect_error):
    base_lr = 0.001
    optim = MockOptimizer(base_lr)
    lr_scheduler = scheduling._CosineAnnealingWithWarmup(
        optim,
        total_steps,
        warmup_steps=warmup_steps,
        eta_min=min_lr,
        error_on_exceeding_steps=expect_error,
    )

    for i in range(total_steps - 1):  # First step is executed on lr_scheduler construction.
        optim.step()
        lr_scheduler.step()
        if i == 0:
            if warmup_steps > 0:
                # Ensure the first update step was performed with a lower lr.
                assert optim.last_lr < base_lr
        elif i < warmup_steps:
            # Learning rate should increase
            assert optim.last_lr < optim.param_groups[0]["lr"]
            assert optim.param_groups[0]["lr"] <= base_lr
        else:
            # Learning rate should decrease
            assert optim.last_lr > optim.param_groups[0]["lr"]
            assert optim.param_groups[0]["lr"] >= min_lr

    optim.step()
    assert optim.last_lr == min_lr

    if expect_error:
        with pytest.raises(ValueError):
            lr_scheduler.step()
    else:
        lr_scheduler.step()
        optim.step()
        assert optim.last_lr == min_lr


@pytest.mark.parametrize(
    "start_step, end_step, start_value,end_value",
    [(10.0, 20.0, 1.0, 2.0), (0.0, 1.0, 1.0, 2.0), (0.0, 100.0, 2.0, 1.0)],
)
def test_linear_hp_scheduler(start_step, end_step, start_value, end_value):
    sched = scheduling.LinearHPScheduler(
        start_step=start_step, end_step=end_step, start_value=start_value, end_value=end_value
    )

    sched.update_global_step(0)
    assert float(sched) == start_value
    sched.update_global_step(start_step - 1)
    assert float(sched) == start_value
    sched.update_global_step(start_step)
    assert float(sched) == start_value
    if start_step + 1 != end_step:
        sched.update_global_step(int((start_step + end_step) / 2))
        assert float(sched) == (start_value + end_value) / 2.0
    sched.update_global_step(end_step)
    assert float(sched) == end_value
    sched.update_global_step(end_step + 1)
    assert float(sched) == end_value
