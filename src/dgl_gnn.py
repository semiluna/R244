""" Code adapted from https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html#PyTorch-Geometric
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv

gnn_layer_by_name = {"GAT": GATConv }

class dgl_GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        c_hidden=16,
        num_layers=3,
        layer_name="GAT",
        dp_rate=0.1,
        num_heads=1,
        **kwargs,
    ):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_feats=in_channels, out_feats=out_channels, num_heads=1, **kwargs),
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_feats=in_channels, out_feats=c_out, num_heads=1, **kwargs)]
        self.layers = nn.ModuleList(layers)
        
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, g, in_feat):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        h = in_feat
        for layer in self.layers[:-1]:
            h = layer(g, h)
            h = self.activation(h)
            h = self.dropout(h)
        h = self.layers[-1](g, h)

        return torch.flatten(h, start_dim=1)