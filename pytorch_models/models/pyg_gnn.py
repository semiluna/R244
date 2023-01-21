""" Code adapted from https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html#PyTorch-Geometric
"""
import torch
import torch.nn as nn

import torch_geometric.nn as geom_nn
import torch.nn.functional as F

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