import torch

from ocl.optimization import OptimizationWrapper


def test_optimization_with_parameter_groups():
    class NestedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.nn.Linear(10, 20)
            self.B = torch.nn.Linear(20, 10)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.nested = NestedModel()
            self.C = torch.nn.Linear(10, 10)

    base_lr = 1e-2
    adapted_lr = 1e-3
    parameter_groups = [
        {"params": ["nested.A", "C"], "momentum": True},
        {
            "params": "nested.B",
            "predicate": lambda name, param: "bias" not in name,
            "lr": adapted_lr,
        },
    ]

    optimization_wrapper = OptimizationWrapper(
        lambda params: torch.optim.SGD(params, lr=base_lr), parameter_groups=parameter_groups
    )

    model = MyModel()
    optimizer = optimization_wrapper(model)["optimizer"]
    assert len(optimizer.param_groups) == len(parameter_groups)

    group1 = optimizer.param_groups[0]
    expected_params = [model.nested.A.weight, model.nested.A.bias, model.C.weight, model.C.bias]
    assert len(group1["params"]) == len(expected_params)
    assert all(p1 is p2 for p1, p2 in zip(group1["params"], expected_params))
    assert group1["lr"] == base_lr
    assert group1["momentum"]

    group2 = optimizer.param_groups[1]
    expected_params = [model.nested.B.weight]
    assert len(group2["params"]) == len(expected_params)
    assert all(p1 is p2 for p1, p2 in zip(group2["params"], expected_params))
    assert group2["lr"] == adapted_lr
    assert not group2["momentum"]
