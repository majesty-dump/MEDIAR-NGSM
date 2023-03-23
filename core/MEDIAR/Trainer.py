import torch
import torch.nn as nn
import numpy as np
import os, sys
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.metrics import CumulativeAverage
import segmentation_models_pytorch as smp
from collections import namedtuple

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.BaseTrainer import BaseTrainer
from core.MEDIAR.utils import *

__all__ = ["Trainer"]
UNDEFINED_LABEL_INDEX = 4
BACKGROUND_LABEL_INDEX = 0


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        dataloaders,
        optimizer,
        scheduler=None,
        criterion=None,
        num_epochs=100,
        device="cuda:0",
        no_valid=False,
        valid_frequency=1,
        amp=False,
        algo_params=None,
    ):
        super(Trainer, self).__init__(
            model,
            dataloaders,
            optimizer,
            scheduler,
            criterion,
            num_epochs,
            device,
            no_valid,
            valid_frequency,
            amp,
            algo_params,
        )

        # Losses 
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True, ignore_index=UNDEFINED_LABEL_INDEX)

        self.MediarLoss = namedtuple('MediarLoss', ['bce_loss', 'mse_loss', 'dice_loss'])

        # Metrics
        self.loss_metric = CumulativeAverage()
        self.mse_loss_metric = CumulativeAverage()
        self.bce_loss_metric = CumulativeAverage()
        self.class_segm_metric = CumulativeAverage()

    def mediar_criterion(self, outputs, labels_onehot_flows, class_labels):
        """loss function between true labels and prediction outputs"""

        # Cell Recognition Loss
        cellprob_loss = self.bce_loss(
            outputs[0][:, -1],
            torch.from_numpy(labels_onehot_flows[:, 1] > 0.5).to(self.device).float(),
        )
        # Cell Distinction Loss
        gradient_flows = torch.from_numpy(labels_onehot_flows[:, 2:]).to(self.device)
        
        gradflow_loss = 0.5 * self.mse_loss(outputs[0][:, :2], 5.0 * gradient_flows)
        class_segm_loss = self.dice_loss(outputs[1], class_labels.long())

        return self.MediarLoss(cellprob_loss, gradflow_loss, class_segm_loss)

    def _epoch_phase(self, phase):
        phase_results = {}

        # Set model mode
        self.model.train() if phase == "train" else self.model.eval()

        # Epoch process
        for batch_data in tqdm(self.dataloaders[phase]):
            images, labels, class_labels = batch_data["img"], batch_data["label"], batch_data["class"]
            if self.with_public:
                # Load batches sequentially from the unlabeled dataloader
                try:
                    batch_data = next(self.public_iterator)
                    images_pub, labels_pub = batch_data["img"], batch_data["label"]

                except:
                    # Assign memory loader if the cycle ends
                    self.public_iterator = iter(self.public_loader)
                    batch_data = next(self.public_iterator)
                    images_pub, labels_pub = batch_data["img"], batch_data["label"]

                # Concat memory data to the batch
                images = torch.cat([images, images_pub], dim=0)
                labels = torch.cat([labels, labels_pub], dim=0)

            images = images.to(self.device)
            labels = labels.to(self.device)
            class_labels = class_labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.amp):
                with torch.set_grad_enabled(phase == "train"):
                    # Output shape is B x [grad y, grad x, cellprob] x H x W

                    outputs = self._inference(images, phase)

                    # Map label masks to graidnet and onehot
                    labels_onehot_flows = labels_to_flows(
                        labels, use_gpu=True, device=self.device
                    )
                    # Calculate loss
                    mediar_loss = self.mediar_criterion(outputs, labels_onehot_flows, class_labels)

                    self.bce_loss_metric.append(mediar_loss.bce_loss)
                    self.mse_loss_metric.append(mediar_loss.mse_loss)
                    self.class_segm_metric.append(mediar_loss.dice_loss)

                    loss = sum(mediar_loss)
                    self.loss_metric.append(loss)
                    # Calculate valid statistics
                    if phase != "train":
                        # TODO: make a separate metric for class segemntation head  
                        outputs, labels = self._post_process(outputs[0], labels)
                        f1_score = self._get_f1_metric(outputs, labels)
                        self.f1_metric.append(f1_score)

                # Backward pass
                if phase == "train":
                    # For the mixed precision training
                    if self.amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    else:
                        loss.backward()
                        self.optimizer.step()

        # Update metrics
        phase_results = self._update_results(phase_results, 
            self.bce_loss_metric, "bce", phase)

        phase_results = self._update_results(phase_results, 
            self.mse_loss_metric, "mse", phase)

        phase_results = self._update_results(phase_results, 
            self.class_segm_metric, "class dl", phase)

        phase_results = self._update_results(phase_results, 
            self.loss_metric, "loss", phase)
        
        if phase != "train":
            phase_results = self._update_results(
                phase_results, self.f1_metric, "f1_score", phase
            )

        return phase_results

    def _inference(self, images, phase="train"):
        """inference methods for different phase"""

        if phase != "train":
            outputs = sliding_window_inference(
                images,
                roi_size=256,
                sw_batch_size=4,
                predictor=self.model,
                padding_mode="constant",
                mode="gaussian",
                overlap=0.5,
            )
        else:
            outputs = self.model(images)
            # print(f"Current phase outputs : {outputs}")
        return outputs

    def _post_process(self, outputs, labels=None):
        """Predict cell instances using the gradient tracking"""
        outputs = outputs.squeeze(0).cpu().numpy()
        gradflows, cellprob = outputs[:2], self._sigmoid(outputs[-1])
        outputs = compute_masks(gradflows, cellprob, use_gpu=True, device=self.device)
        outputs = outputs[0]  # (1, C, H, W) -> (C, H, W)

        if labels is not None:
            labels = labels.squeeze(0).squeeze(0).cpu().numpy()

        return outputs, labels

    def _sigmoid(self, z):
        """Sigmoid function for numpy arrays"""
        return 1 / (1 + np.exp(-z))
