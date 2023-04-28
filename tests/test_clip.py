# Test Clip loss and its functionality in multi GPU training.
import pytest
import pytorch_lightning as pl
import torch

from ocl.losses import CLIPLoss


class ExampleModel(pl.LightningModule):
    def __init__(self, normalize_inputs, learn_scale):
        super().__init__()
        self.clip_loss = CLIPLoss(
            normalize_inputs=normalize_inputs, learn_scale=learn_scale, model_path=None
        )
        self.automatic_optimization = False
        self.last_loss = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def forward(self, representation_a, representation_b):
        return self.clip_loss(representation_a, representation_b, model=self)

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch[0], batch[1])
        # Gather and aggregate loss.
        loss = self.all_gather(loss).mean()
        self.last_loss = loss.detach()
        return loss


def get_data(n_instances, seed):
    generator = torch.random.manual_seed(seed)
    rep_a = torch.randn((n_instances, 256), generator=generator)
    rep_b = torch.randn((n_instances, 256), generator=generator)
    return list(zip(rep_a, rep_b))


@pytest.mark.skip("Involves multiprocessing which is incompatible with CI pipeline.")
@pytest.mark.parametrize("normalize_inputs", [True, False])
def test_loss_consistency(normalize_inputs):
    model = ExampleModel(normalize_inputs, True)
    trainer_single = pl.Trainer(devices=1, max_epochs=1, logger=False)
    data = get_data(10, 879342378)

    dataloader = torch.utils.data.DataLoader(data, batch_size=10)
    trainer_single.fit(model, dataloader)
    single_loss = model.last_loss

    dataloader = torch.utils.data.DataLoader(data, batch_size=2)
    trainer_multi = pl.Trainer(devices=5, strategy="ddp_spawn", max_epochs=1, logger=False)
    trainer_multi.fit(model, dataloader)
    multi_loss = model.last_loss

    assert single_loss == multi_loss
