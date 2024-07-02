import copy

import torch
import torch.nn as nn
import torchvision

import lightly

from lightly.utils import BenchmarkModule
from lightly.models import utils
from lightly.models.modules import heads

class MoCoModel(BenchmarkModule):
    def __init__(
            self, 
            dataloader_kNN, 
            num_classes,
            max_epochs,
            backbone,
            num_ftrs,
            knn_k,
            knn_t,
            lr_factor,
            distributed,
            sync_batchnorm,
            memory_bank_size
        ):
        super().__init__(
            dataloader_kNN, num_classes, knn_k, knn_t
        )


        # create a ResNet backbone and remove the classification head
        num_splits = 0 if sync_batchnorm else 8
        
        # # TODO: Add split batch norm to the resnet model
        # resnet = torchvision.models.resnet18()
        # feature_dim = list(resnet.children())[-1].in_features
        # self.backbone = nn.Sequential(
        #     *list(resnet.children())[:-1],
        #     nn.AdaptiveAvgPool2d(1)
        # )
        self.backbone = backbone

        # create a moco model based on ResNet
        self.projection_head = heads.MoCoProjectionHead(num_ftrs, 2048, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.07,
            memory_bank_size=memory_bank_size)

        self.max_epochs = max_epochs
        self.distributed = distributed
        self.lr_factor = lr_factor

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        # update momentum
        utils.update_momentum(self.backbone, self.backbone_momentum, 0.999)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, 0.999)

        def step(x0_, x1_):
            x1_, shuffle = utils.batch_shuffle(x1_, distributed=self.distributed)
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            x0_ = self.projection_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            x1_ = utils.batch_unshuffle(
                    x1_, shuffle, distributed=self.distributed)
            
            return x0_, x1_

        # We use a symmetric loss (model trains faster at little compute overhead)
        loss_1 = self.criterion(*step(x0, x1))
        loss_2 = self.criterion(*step(x1, x0))

        loss = 0.5 * (loss_1 + loss_2)

        self.log('train_loss_ssl', loss)
        self.log('val_max_accuracy', self.max_accuracy * 100.0)

        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + list(self.projection_head.parameters())
        optim = torch.optim.SGD(
            params, 
            lr=0.03 * self.lr_factor,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, self.max_epochs)
        
        return [optim], [scheduler]
    