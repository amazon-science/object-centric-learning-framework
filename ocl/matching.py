"""Methods for matching between sets of elements."""
from typing import Tuple, Type

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchtyping import TensorType

# Avoid errors due to flake:
batch_size = None
n_elements = None

CostMatrix = Type[TensorType["batch_size", "n_elements", "n_elements"]]
AssignmentMatrix = Type[TensorType["batch_size", "n_elements", "n_elements"]]
CostVector = Type[TensorType["batch_size"]]


class Matcher(torch.nn.Module):
    """Matcher base class to define consistent interface."""

    def forward(self, C: CostMatrix) -> Tuple[AssignmentMatrix, CostVector]:
        pass


class CPUHungarianMatcher(Matcher):
    """Implementaiton of a cpu hungarian matcher using scipy.optimize.linear_sum_assignment."""

    def forward(self, C: CostMatrix) -> Tuple[AssignmentMatrix, CostVector]:
        X = torch.zeros_like(C)
        C_cpu: np.ndarray = C.detach().cpu().numpy()
        for i, cost_matrix in enumerate(C_cpu):
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            X[i][row_ind, col_ind] = 1.0
        return X, (C * X).sum(dim=(1, 2))
