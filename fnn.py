import hydra
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from omegaconf import DictConfig
from torch.nn.functional import cross_entropy, one_hot
from torch.optim import Adam
from torchmetrics import Accuracy

from mnist_datamodule import MNISTDataModule


class SimpleFFN(LightningModule):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        input2hidden_size = hidden_sizes[0]
        hidden_size = hidden_sizes[1]  # TODO: dynamic hidden layer num
        hidden2output_size = hidden_sizes[-1]

        self.model = nn.Sequential(
            nn.Linear(input_size, input2hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden2output_size, output_size),
            # nn.Softmax()
        )

        self.learning_rate = learning_rate

        self.loss_module = nn.CrossEntropyLoss()

        self.acc = Accuracy()

    def forward(self, x):
        return self.model(x.view(-1, self.input_size))

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_module(y_hat, one_hot(y, num_classes=self.output_size).float())
        self.acc(y_hat.argmax(dim=1), y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_module(y_hat, one_hot(y, num_classes=self.output_size).float())
        self.acc(y_hat.argmax(dim=1), y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.acc, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    wandb_logger = WandbLogger(
        "simple_ffn"
    )

    model = SimpleFFN(
        28*28,
        [512, 512, 512],
        10
    )

    mnist_datamodule = MNISTDataModule('.', batch_size=128)
    mnist_datamodule.prepare_data()
    mnist_datamodule.setup()

    checkpoint_callback = ModelCheckpoint(dirpath="model_checkpoints", save_top_k=3, monitor="val_loss")

    trainer = Trainer(
        callbacks=[checkpoint_callback],
        auto_lr_find=True,
        accelerator='gpu',
        max_epochs=100,
        logger=wandb_logger
    )

    # trainer.tune(model, datamodule=mnist_datamodule)

    trainer.fit(model, datamodule=mnist_datamodule)

    print(f"Best model ({checkpoint_callback.best_model_score}) at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
