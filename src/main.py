import os
import argparse

import torch

from torch.utils.data import DataLoader

import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
from torch_geometric.datasets import Reddit as pyg_Reddit

from dgl.data import RedditDataset as dgl_Reddit 
from dgl.dataloading import Sampler#, DataLoader
# PL callbacks
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

from gnn_trainer import pyg_NodeLevelGNN, dgl_NodeLevelGNN

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/GNNs/")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class SubgraphSampler(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self, g, indices):
        return g

def collate_fn(data):
    graph = data[0]
    return graph, graph.ndata['feat']
    
# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print("Train accuracy: %4.2f%%" % (100.0 * result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100.0 * result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100.0 * result_dict["test"]))

def train_node_classifier(model_name, dataset, mode, **model_kwargs):
    pl.seed_everything(42)
    if mode == 'pyg':
        node_data_loader = geom_data.DataLoader(dataset, batch_size=1)
    else:
        assert mode == 'dgl', 'Support for other GNN libraries is not implemented'
        num_nodes = dataset[0].ndata['feat'].shape[0]
        # sampler = SubgraphSampler()
        node_data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        num_node_features = dataset[0].ndata['feat'].shape[-1]

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        gpus=AVAIL_GPUS,
        max_epochs=200,
    )  # 0 because epoch size is 1
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(CHECKPOINT_PATH, "NodeLevel%s.ckpt" % model_name)
    # if os.path.isfile(pretrained_filename):
    #     print("Found pretrained model, loading...")
    #     model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    # else:
    pl.seed_everything()
    
    if mode == 'pyg':
        model = pyg_NodeLevelGNN(
            model_name=model_name, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs
        )
        trainer.fit(model, node_data_loader, node_data_loader)
        model = pyg_NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        # import ipdb; ipdb.set_trace()
        model = dgl_NodeLevelGNN(
            model_name=model_name, c_in=num_node_features, c_out=dataset.num_classes, **model_kwargs
        )
        trainer.fit(model, node_data_loader, node_data_loader)
        model = dgl_NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result

def pyg_main():
    dataset = pyg_Reddit(root='pyg_reddit/')
    _, result = train_node_classifier('GAT', dataset, 'pyg')
    print_results(result)

def dgl_main():
    dataset = dgl_Reddit(self_loop=True, raw_dir='dgl_reddit/')
    _, result = train_node_classifier('GAT', dataset, 'dgl')
    print_results(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['pyg', 'dgl'])
    args = parser.parse_args()
    if args.mode == 'pyg':
        pyg_main()
    else:
        assert args.mode == 'dgl'
        dgl_main()