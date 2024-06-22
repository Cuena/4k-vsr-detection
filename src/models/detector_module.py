from typing import Any, Dict, Tuple, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from utils.image_utils import get_2d_dct
from sklearn.metrics import accuracy_score

class DetectorModule(LightningModule):
    
    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        mlp: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        concat_dct_features: bool=False,
        methods: List[str]=None,
        all_methods: List[str]=None,
    ) -> None:
       
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # self.save_hyperparameters(logger=False, ignore=["feature_extractor", "classifier"])

        self.feature_extractor = feature_extractor
        self.mlp = mlp

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        self.methods = methods

        self.num_classes = len(methods)

        self.concat_dct_features = concat_dct_features
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        
        self.predictions = []
        self.targets = []

    def log_individual_classes(self, outs, mode):
        # Extract all predictions and labels from the validation step
        if mode == "train":
            preds = [out['preds'] for out in outs]
            labels = [out['y'] for out in outs]
        else:
            _, preds, labels, _ = zip(*outs)

        # Combine all predictions and labels into a single tensor
        overall_preds = torch.cat(preds)
        overall_labels = torch.cat(labels)

        for class_id in range(self.num_classes):
            class_indices = overall_labels == class_id
            class_preds = overall_preds[class_indices]
            class_labels = overall_labels[class_indices]

            if class_labels.shape[0] > 0:
                class_accuracy = accuracy_score(class_labels.cpu(), class_preds.cpu())

            else:
                print(f"No samples for class {class_id} in this batch")
                class_accuracy = 0.0

            self.log(
                f"{mode}/acc_{self.methods[class_id]}", class_accuracy, prog_bar=True
            )

        if not self.trainer.sanity_checking:
            if self.log_wrong_predictions_every is not None and isinstance(
                    self.log_wrong_predictions_every, int
            ):
                if (self.current_epoch + 1) % self.log_wrong_predictions_every == 0:
                    self.log_wrong_predictions(outs)

    def concat_dct_features(self, x: torch.Tensor) -> torch.Tensor:
        dct_features = get_2d_dct(x)
        x = torch.cat([x, dct_features], dim=1)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        x = self.feature_extractor(x)
        if self.concat_dct_features:
            x = self.concat_dct_features(x)
        x = self.mlp(x)
        return x

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y, _ = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.predictions.append(preds)
        self.targets.append(targets)

        # return loss or backpropagation will fail
        return loss


    def log_individual_classes(self, predictions, targets, mode):
   
        for class_id in range(self.num_classes):
            class_indices = targets == class_id
            class_preds = predictions[class_indices]
            class_labels = targets[class_indices]

            if class_labels.shape[0] > 0:
                class_accuracy = accuracy_score(class_labels.cpu(), class_preds.cpu())

            else:
                print(f"No samples for class {class_id} in this batch")
                class_accuracy = 0.0

            self.log(
                f"{mode}/acc_{self.methods[class_id]}", class_accuracy, prog_bar=True
            )

        # if not self.trainer.sanity_checking:
        #     if self.log_wrong_predictions_every is not None and isinstance(
        #             self.log_wrong_predictions_every, int
        #     ):
        #         if (self.current_epoch + 1) % self.log_wrong_predictions_every == 0:
        #             self.log_wrong_predictions(outs)

    def on_train_epoch_end(self) -> None:
        
        epoch_predictions = torch.cat(self.predictions)
        epoch_targets = torch.cat(self.targets)
        self.log_individual_classes(epoch_predictions, epoch_targets, "train")
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.feature_extractor = torch.compile(self.feature_extractor)
            self.mlp = torch.compile(self.mlp)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = DetectorModule(None, None, None, None)
