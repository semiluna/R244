import os
import argparse

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
from torch_geometric.datasets import Reddit as pyg_Reddit, Planetoid

from dgl.data import RedditDataset as dgl_Reddit 
from dgl.dataloading import Sampler

import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

dgl_DATASET = {
    'cora': CoraGraphDataset,
    'citeseer': CiteseerGraphDataset,
    'pubmed': PubmedGraphDataset,
    'reddit': dgl_Reddit,
}

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/pytorch/")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class pyg_GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        c_hidden=16,
        num_layers=3,
        dp_rate=0.6,
        **kwargs,
    ):
        super().__init__()

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                geom_nn.GATConv(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    heads=1, 
                    dropout=0.6,
                    **kwargs),
                nn.ELU(),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [
            geom_nn.GATConv(
                in_channels=in_channels, 
                out_channels=c_out, **kwargs)
            ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

class pyg_NodeLevelGNN(pl.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        
        self.model = pyg_GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Unknown forward mode: %s" % mode

        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3, weight_decay=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)

class dgl_GNNModel(nn.Module):
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

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h

class dgl_NodeLevelGNN(pl.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        
        self.model = dgl_GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        g, features = data[0], data[1]
        labels = g.ndata['label']

        x = self.model(g, features)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = g.ndata['train_mask']
        elif mode == "val":
            mask = g.ndata['val_mask']
        elif mode == "test":
            mask = g.ndata['test_mask']
        else:
            assert False, "Unknown forward mode: %s" % mode

        loss = self.loss_module(x[mask], labels[mask])

        logits = x[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return loss, correct.item() * 1.0 / len(labels)

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3, weight_decay=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)


class SubgraphSampler(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self, g, indices):
        return g

def collate_fn(data):
    graph = data[0]
    return graph, graph.ndata['feat']
    
# Small function for printing the test scores
def print_results(result_dict, timed, args):
    if "train" in result_dict:
        print("Train accuracy: %4.2f%%" % (100.0 * result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100.0 * result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100.0 * result_dict["test"]))
    print(f"TRAINING TIME (s): {timed}")
    with open('statistics.csv', 'a+') as handle:
        print(f'{args.mode},{args.gpus},{args.strategy},{timed},', file=handle)

def train_node_classifier(model_name, dataset, mode, **model_kwargs):
    pl.seed_everything(42)
    if mode == 'pyg':
        node_data_loader = geom_data.DataLoader(dataset, batch_size=1)
    else:
        assert mode == 'dgl', 'Support for other GNN libraries is not implemented'
        num_nodes = dataset[0].ndata['feat'].shape[0]
        dataset[0].add_edges(list(range(num_nodes)), list(range(num_nodes))) # add self loops so GAT does not have problems

        # sampler = SubgraphSampler()
        node_data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        num_node_features = dataset[0].ndata['feat'].shape[-1]

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    if args.gpus > 0:
        trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
            gpus=AVAIL_GPUS,
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
    
    if mode == 'pyg':
        model = pyg_NodeLevelGNN(
            model_name=model_name, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs
        )
        training_start = time.time()
        trainer.fit(model, node_data_loader, node_data_loader)
        training_end = time.time()
        model = pyg_NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        # import ipdb; ipdb.set_trace()
        model = dgl_NodeLevelGNN(
            model_name=model_name, in_size=num_node_features, out_size=dataset.num_classes, **model_kwargs
        )
        training_start = time.time()
        trainer.fit(model, node_data_loader, node_data_loader)
        training_end = time.time()
        model = dgl_NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    # batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result, training_end - training_start

def pyg_main(args):
    if args.dataset == 'reddit':
        dataset = pyg_Reddit(root='pyg_reddit/')
    else:
        dataset = Planetoid(f'pyg_{args.dataset}', args.dataset)
        
    _, result, training_time = train_node_classifier('GAT', dataset, 'pyg')
    print_results(result, training_time, args)

def dgl_main(args):
    dataset_cls = dgl_DATASET[args.dataset]
    dataset = dataset_cls(raw_dir=f'dgl_{args.dataset}/', transform=AddSelfLoop())
    _, result, training_time = train_node_classifier('GAT', dataset, 'dgl')
    print_results(result, training_time, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['cora', 'citeseer', 'pubmed', 'reddit'], default='cora')
    parser.add_argument('--mode', type=str, choices=['pyg', 'dgl'])
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--strategy', type=str, default='ddp')
    parser.add_argument('--epochs', type=int, default=200)

    args = parser.parse_args()
    if args.mode == 'pyg':
        pyg_main(args)
    else:
        assert args.mode == 'dgl'
        dgl_main(args)