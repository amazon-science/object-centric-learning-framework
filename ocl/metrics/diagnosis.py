"""Metrics used for diagnosis."""
import torch
import torchmetrics


class TensorStatistic(torchmetrics.Metric):
    """Metric that computes summary statistic of tensors for logging purposes.

    First dimension of tensor is assumed to be batch dimension. Other dimensions are reduced to a
    scalar by the chosen reduction approach (sum or mean).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError(f"Unknown reduction {reduction}")
        self.reduction = reduction
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, tensor: torch.Tensor):
        tensor = torch.atleast_2d(tensor).flatten(1, -1).to(dtype=torch.float64)

        if self.reduction == "mean":
            tensor = torch.mean(tensor, dim=1)
        elif self.reduction == "sum":
            tensor = torch.sum(tensor, dim=1)

        self.values += tensor.sum()
        self.total += len(tensor)

    def compute(self) -> torch.Tensor:
        return self.values / self.total
