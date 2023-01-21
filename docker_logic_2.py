import os
import os.path as osp
import time
import argparse
import copy

import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_geometric.nn as geom_nn
from torch_geometric.datasets import Reddit as pyg_Reddit
from torch_geometric.loader import NeighborLoader

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

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/")

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

class dgl_GAT(nn.Module):
    def __init__(self, in_size, out_size, hid_size=16, heads=[1,1]):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[0],
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )

        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.gat_layers, blocks)):
            h = layer(block, h)
            if l == len(self.gat_layers) - 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h        

class pyg_SAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=256):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(geom_nn.SAGEConv(in_channels, hidden_channels))
        self.convs.append(geom_nn.SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(geom_nn.SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

class pyg_GAT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=16,
        num_layers=3,
        dp_rate=0.6,
        **kwargs,
    ):
        super().__init__()

        layers = []
        in_c, out_c = in_channels, hidden_channels
        for l_idx in range(num_layers - 1):
            layers += [
                geom_nn.GATConv(
                    in_channels=in_c, 
                    out_channels=out_c, 
                    heads=1, 
                    dropout=0.6,
                    **kwargs),
                nn.ELU(),
                nn.Dropout(dp_rate),
            ]
            in_c = hidden_channels
        layers += [
            geom_nn.GATConv(
                in_channels=in_c, 
                out_channels=out_channels, **kwargs)
            ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class dgl_SAGETrainer(pl.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        if model_name == 'sage':
            self.model = dgl_SAGE(**model_kwargs)
        else:
            self.model = dgl_GAT(**model_kwargs)

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

class pyg_DataModule(pl.LightningDataModule):
    def __init__(self, args, path):
        self.path = path 
        self.args = args
        self.train_batch = args.train_batch
        self.val_batch = args.val_batch
        self.data_workers = args.data_workers
        self.prepare_data_per_node = False
        self._log_hyperparams = False

    def prepare_data(self):
        pyg_Reddit(self.path)

    def setup(self, stage):
        self.dataset = pyg_Reddit(self.path)
        self.data = self.dataset[0]

    def train_dataloader(self):
        kwargs = {'batch_size': self.train_batch, 'num_workers': self.data_workers}
        return NeighborLoader(self.data, input_nodes=self.data.train_mask,
                                num_neighbors=[10, 10, 10], shuffle=True, **kwargs)
    def val_dataloader(self):
        kwargs = {'batch_size': self.val_batch, 'num_workers': self.data_workers}
        validation_loader = NeighborLoader(copy.copy(self.data), input_nodes=self.data.val_mask,
                                    num_neighbors=[10, 10, 10], shuffle=False, **kwargs)

        validation_loader.data.num_nodes = self.data.num_nodes
        validation_loader.data.n_id = torch.arange(self.data.num_nodes)   

        return validation_loader
    
    def test_dataloader(self):
        return self.val_dataloader()


class pyg_SAGETrainer(pl.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        if model_name == 'sage':
            self.model = pyg_SAGE(**model_kwargs)
        else:
            self.model = pyg_GAT(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, batch):

        # (input_nodes, output_nodes, blocks) = data
        # x = blocks[0].srcdata["feat"]
        # y = blocks[-1].dstdata["label"]
        # y_hat = self.model(blocks, x)
        # loss = F.cross_entropy(y_hat, y)
        # acc = (y_hat.argmax(dim=-1) == y).sum().float() / len(y_hat)
        # return loss, acc, len(y_hat)

        y = batch.y[:batch.batch_size]
        y_hat = self.model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        
        total_correct = int((y_hat.argmax(dim=-1) == y).sum())
        total_examples = batch.batch_size

        return loss, total_correct / total_examples, total_examples

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


def pyg_mini_batching(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'Reddit')
    dataset = pyg_Reddit(path)

    reddit_module = pyg_DataModule(args, path)

    num_nodes = dataset.num_node_features
    num_node_features = dataset.num_node_features

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

    model = pyg_SAGETrainer(args.model, in_channels=num_node_features, out_channels=dataset.num_classes)
    training_start = time.time()
    trainer.fit(model, datamodule=reddit_module)
    training_end = time.time()
    # model = pyg_SAGETrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # trainer.test(model, datamodule=reddit_module)
    return training_end - training_start
    

def dgl_mini_batching(args):
    # dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root='ogbn_products'))
    dataset = AsNodePredDataset(dgl.data.RedditDataset(self_loop=True))
    g = dataset[0]
    num_nodes = dataset[0].ndata['feat'].shape[0]
    num_node_features = dataset[0].ndata['feat'].shape[-1]
    # avoid creating certain graph formats in each sub-process to save momory
    
    g.create_formats_()
    print('Created graph formats........')
    # pin to CPU for faster access
    # if args.gpus > 0:
    #     g.pin_memory_()

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
    print('Start training....')
    model = dgl_SAGETrainer(args.model, in_size=num_node_features, out_size=dataset.num_classes)
    training_start = time.time()
    trainer.fit(model, train_dataloader, val_dataloader)
    training_end = time.time()
    # model = dgl_SAGETrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # trainer.test(model, dataloaders=val_dataloader, verbose=False)
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
    parser.add_argument('--model', type=str, choices=['sage', 'gat'])

    args = parser.parse_args()
    assert args.model in ['sage', 'gat']
    
    if args.mode == 'dgl':
        training_time = dgl_mini_batching(args)
    else:
        training_time = pyg_mini_batching(args)

    print_results(training_time, args)

