from typing import Union, List

import hydra
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import nn, stack
from omegaconf import DictConfig
from torch.nn.functional import one_hot
from torch.optim import Adam
from torchmetrics import Accuracy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
        return {
            "loss": loss,
            "preds": y_hat.argmax(dim=1),
            "labels": y,
        }

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        return self._epoch_end_logging(outputs, 'train')

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_module(y_hat, one_hot(y, num_classes=self.output_size).float())
        self.acc(y_hat.argmax(dim=1), y)

        # self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", self.acc, prog_bar=True)
        return {
            "loss": loss,
            "labels": y,
            "preds": y_hat.argmax(dim=1),
        }

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        return self._epoch_end_logging(outputs, 'val')

    def _epoch_end_logging(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]], prefix: str) -> None:
        avg_loss = stack([x["loss"] for x in outputs]).mean()
        self.log(f"{prefix}_loss", avg_loss)

        preds = stack([x["preds"] for x in outputs]).flatten().cpu().numpy()
        labels = stack([x["labels"] for x in outputs]).flatten().cpu().numpy()

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="macro")
        recall = recall_score(labels, preds, average="macro")
        f1 = f1_score(labels, preds, average="macro")

        self.log(f"{prefix}_accuracy", accuracy)
        self.log(f"{prefix}_precision", precision)
        self.log(f"{prefix}_recall", recall)
        self.log(f"{prefix}_f1", f1)

        digit_precision = precision_score(labels, preds, average=None)
        digit_recall = recall_score(labels, preds, average=None)
        digit_f1 = f1_score(labels, preds, average=None)

        for i in range(10):
            self.log(f"{prefix}_precision/{i}", digit_precision[i])
            self.log(f"{prefix}_recall/{i}", digit_recall[i])
            self.log(f"{prefix}_f1/{i}", digit_f1[i])

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    wandb_logger = WandbLogger(
        entity="dpis-disciples",
        project='ml2-project-ffn'
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
        accelerator='gpu',  # 'gpu',
        devices=[0],
        max_epochs=100,
        logger=wandb_logger
    )

    # trainer.tune(model, datamodule=mnist_datamodule)

    trainer.fit(model, datamodule=mnist_datamodule)

    print(f"Best model ({checkpoint_callback.best_model_score}) at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
