"""Implementation of conditioning approaches for slots."""
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn

import ocl.typing


class RandomConditioning(nn.Module):
    """Random conditioning with potentially learnt mean and stddev."""

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        learn_mean: bool = True,
        learn_std: bool = True,
        mean_init: Callable[[torch.Tensor], None] = torch.nn.init.xavier_uniform_,
        logsigma_init: Callable[[torch.Tensor], None] = nn.init.xavier_uniform_,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = object_dim

        if learn_mean:
            self.slots_mu = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_mu", torch.zeros(1, 1, object_dim))

        if learn_std:
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_logsigma", torch.zeros(1, 1, object_dim))

        with torch.no_grad():
            mean_init(self.slots_mu)
            logsigma_init(self.slots_logsigma)

    def forward(self, batch_size: int) -> ocl.typing.ConditioningOutput:
        mu = self.slots_mu.expand(batch_size, self.n_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, self.n_slots, -1)
        return mu + sigma * torch.randn_like(mu)


class LearntConditioning(nn.Module):
    """Conditioning with a learnt set of slot initializations, similar to DETR."""

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        slot_init: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        """Initialize LearntConditioning.

        Args:
            object_dim: Dimensionality of the conditioning vector to generate.
            n_slots: Number of conditioning vectors to generate.
            slot_init: Callable used to initialize individual slots.
        """
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = object_dim

        self.slots = nn.Parameter(torch.zeros(1, n_slots, object_dim))

        if slot_init is None:
            slot_init = nn.init.normal_

        with torch.no_grad():
            slot_init(self.slots)

    def forward(self, batch_size: int) -> ocl.typing.ConditioningOutput:
        """Generate conditioining vectors for `batch_size` instances.

        Args:
            batch_size: Number of instances to create conditioning vectors for.

        Returns:
            The conditioning vectors.
        """
        return self.slots.expand(batch_size, -1, -1)


class RandomConditioningWithQMCSampling(RandomConditioning):
    """Random gaussian conditioning using Quasi-Monte Carlo (QMC) samples."""

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        learn_mean: bool = True,
        learn_std: bool = True,
        mean_init: Callable[[torch.Tensor], None] = torch.nn.init.xavier_uniform_,
        logsigma_init: Callable[[torch.Tensor], None] = torch.nn.init.xavier_uniform_,
    ):
        """Initialize RandomConditioningWithQMCSampling.

        Args:
            object_dim: Dimensionality of the conditioning vector to generate.
            n_slots: Number of conditioning vectors to generate.
            learn_mean: Learn the mean vector of sampling distribution.
            learn_std: Learn the std vector for sampling distribution.
            mean_init: Callable to initialize mean vector.
            logsigma_init: Callable to initialize logsigma.
        """
        super().__init__(
            object_dim=object_dim,
            n_slots=n_slots,
            learn_mean=learn_mean,
            learn_std=learn_std,
            mean_init=mean_init,
            logsigma_init=logsigma_init,
        )

        import scipy.stats  # Import lazily because scipy takes some time to import

        self.randn_rng = scipy.stats.qmc.MultivariateNormalQMC(mean=np.zeros(object_dim))

    def _randn(self, *args: Tuple[int]) -> torch.Tensor:
        n_elements = np.prod(args)
        # QMC sampler needs to sample powers of 2 numbers at a time
        n_elements_rounded2 = 2 ** int(np.ceil(np.log2(n_elements)))
        z = self.randn_rng.random(n_elements_rounded2)[:n_elements]

        return torch.from_numpy(z).view(*args, -1)

    def forward(self, batch_size: int) -> ocl.typing.ConditioningOutput:
        """Generate conditioning vectors for `batch_size` instances.

        Args:
            batch_size: Number of instances to create conditioning vectors for.

        Returns:
            The conditioning vectors.
        """
        mu = self.slots_mu.expand(batch_size, self.n_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, self.n_slots, -1)

        z = self._randn(batch_size, self.n_slots).to(mu, non_blocking=True)
        return mu + sigma * z


class SlotwiseLearntConditioning(nn.Module):
    """Random conditioning with learnt mean and stddev for each slot.

    Removes permutation equivariance compared to the original slot attention conditioning.
    """

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        mean_init: Callable[[torch.Tensor], None] = torch.nn.init.normal_,
        logsigma_init: Callable[[torch.Tensor], None] = torch.nn.init.xavier_uniform_,
    ):
        """Initialize SlotwiseLearntConditioning.

        Args:
            object_dim: Dimensionality of the conditioning vector to generate.
            n_slots: Number of conditioning vectors to generate.
            mean_init: Callable to initialize mean vector.
            logsigma_init: Callable to initialize logsigma.
        """
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = object_dim

        self.slots_mu = nn.Parameter(torch.zeros(1, n_slots, object_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, n_slots, object_dim))

        with torch.no_grad():
            mean_init(self.slots_mu)
            logsigma_init(self.slots_logsigma)

    def forward(self, batch_size: int) -> ocl.typing.ConditioningOutput:
        """Generate conditioning vectors for `batch_size` instances.

        Args:
            batch_size: Number of instances to create conditioning vectors for.

        Returns:
            The conditioning vectors.
        """
        mu = self.slots_mu.expand(batch_size, -1, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, -1, -1)
        return mu + sigma * torch.randn_like(mu)
