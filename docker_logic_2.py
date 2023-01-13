import os
import time
import argparse

import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm

import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.multiprocessing import shared_tensor

from ogb.nodeproppred import DglNodePropPredDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/dgl/")

class dgl_SAGE(nn.Module):
    def __init__(self, in_size, out_size, hid_size=256):
        super().__init__()
        self.hidden_features = hid_size
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, use_uva):
        g.ndata["h"] = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])
        for l, layer in enumerate(self.layers):
            dataloader = DataLoader(
                g,
                torch.arange(g.num_nodes(), device=device),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=use_uva,
            )
            # in order to prevent running out of GPU memory, allocate a
            # shared output tensor 'y' in host memory
            y = shared_tensor(
                (
                    g.num_nodes(),
                    self.hid_size
                    if l != len(self.layers) - 1
                    else self.out_size,
                )
            )
            for input_nodes, output_nodes, blocks in (
                tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
            ):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # non_blocking (with pinned memory) to accelerate data transfer
                y[output_nodes] = h.to(y.device, non_blocking=True)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            g.ndata["h"] = y if use_uva else y.to(device)

        g.ndata.pop("h")
        return y

class dgl_SAGETrainer(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        
        self.model = dgl_SAGE(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data):

        (input_nodes, output_nodes, blocks) = data
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
        y_hat = self.model(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).sum().float() / len(y_hat)
        return loss, acc, len(y_hat)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3, weight_decay=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.forward(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc, sz = self.forward(batch)
        return {'acc': acc, 'len': sz}

    def validation_epoch_end(self, outputs):
        total_n = 0.0
        total_acc = 0.0
        for output in outputs:
            total_acc += output['acc'] * output['len']
            total_n += output['len']
        self.log('val_acc', total_acc / total_n)
        return {'val_acc': total_acc / total_n}
    
    def test_step(self, batch, batch_idx):
        _, acc, sz = self.forward(batch)
        return {'test_acc': acc, 'len': sz}

    def test_epoch_end(self, outputs):
        total_n = 0.0
        total_acc = 0.0
        for output in outputs:
            total_acc += output['test_acc'] * output['len']
            total_n += output['len']
        self.log('test_acc', total_acc / total_n)
        print("TEST ACCURACY:  {:4.2f}".format( 100 * (total_acc / total_n)))


def dgl_mini_batching(args):
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root='ogbn_products'))
    g = dataset[0]
    num_nodes = dataset[0].ndata['feat'].shape[0]
    num_node_features = dataset[0].ndata['feat'].shape[-1]
    # avoid creating certain graph formats in each sub-process to save momory
    g.create_formats_()

    # pin to CPU for faster access
    if args.gpus > 0:
        g.pin_memory_()

    sampler = NeighborSampler(
        [10, 10, 10], prefetch_node_feats=["feat"], prefetch_labels=["label"]
    )
    train_dataloader = DataLoader(
        g,
        dataset.train_idx,
        sampler,
        batch_size=args.train_batch,
        shuffle=True,
        drop_last=False,
        num_workers=args.data_workers,
    )
    val_dataloader = DataLoader(
        g,
        dataset.val_idx,
        sampler,
        batch_size=args.val_batch,
        shuffle=True,
        drop_last=False,
        num_workers=args.data_workers,
    )

    root_dir = os.path.join(CHECKPOINT_PATH, "dgl-mini-batching-sage")
    os.makedirs(root_dir, exist_ok=True)

    # CREATE PL TRAINER
    if args.gpus > 0:
        trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
            max_epochs=args.epochs,
            accelerator='gpu',
            devices=args.gpus,
            strategy=args.strategy,
        )  
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    else:
        trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
            # gpus=AVAIL_GPUS,
            max_epochs=args.epochs) 
    
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    pl.seed_everything()

    model = dgl_SAGETrainer(in_size=num_node_features, out_size=dataset.num_classes)
    training_start = time.time()
    trainer.fit(model, train_dataloader, val_dataloader)
    training_end = time.time()
    model = dgl_SAGETrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    trainer.test(model, dataloaders=val_dataloader, verbose=False)
    return training_end - training_start


def print_results(timed, args):
    # if "train" in result_dict:
    #     print("Train accuracy: %4.2f%%" % (100.0 * result_dict["train"]))
    # if "val" in result_dict:
    #     print("Val accuracy:   %4.2f%%" % (100.0 * result_dict["val"]))
    # print("Test accuracy:  %4.2f%%" % (100.0 * result_dict["test"]))
    print(f"TRAINING TIME (s): {timed}")
    with open('statistics.csv', 'a+') as handle:
        print(f'{args.mode},{args.gpus},{args.strategy},{timed},', file=handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['cora', 'citeseer', 'pubmed', 'reddit'], default='cora')
    parser.add_argument('--mode', type=str, choices=['pyg', 'dgl'])
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--strategy', type=str, default='ddp')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--train_batch', type=int, default=64)
    parser.add_argument('--val_batch', type=int, default=64)
    parser.add_argument('--data_workers', default=0, type=int)

    args = parser.parse_args()

    training_time = dgl_mini_batching(args)
    print_results(training_time, args)

