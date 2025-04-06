import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_attention_layers: int = 3,
        num_heads: int = 8,
    ):
        """
        Initalizes the GraphEncoder

        Args:
            node_input_dim (int): Feature Dimension of input nodes
            embedding_dim (int): Number of dimensions in the embedding space.
            hidden_dim (int): Number of neurons of the hidden layer of the fcl.
            num_attention_layers (int): Number of attention layers.
            num_heads (int): Number of heads in each attention layer
        """
        super().__init__()
        # initial embeds ff layer for each nodes type
        # self.depot_embed = nn.Linear(depot_input_dim, embedding_dim)
        self.node_embed = nn.Linear(node_input_dim, embedding_dim)

        self.attention_layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_attention_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the node embedding for each node
        in each graph.

        Args:
            x (torch.Tensor): Shape (num_graphs, num_nodes, num_features)

        Returns:
            torch.Tensor: Returns the embedding of each node in each graph.
                Shape (num_graphs, num_nodes, embedding_dim).
        """

        out = self.node_embed(x)
        for layer in self.attention_layers:
            out = layer(out)

        return out


class GraphDemandEncoder(GraphEncoder):
    def __init__(
        self,
        depot_input_dim: int,
        node_input_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_attention_layers: int = 3,
        num_heads: int = 8,
    ):
        """
        Initalizes the GraphDemandEncoder

        Args:
            depot_input_dim (int): Feature Dimension of input depots
            node_input_dim (int): Feature Dimension of input nodes
            embedding_dim (int): Number of dimensions in the embedding space.
            hidden_dim (int): Number of neurons of the hidden layer of the fcl.
            num_attention_layers (int): Number of attention layers.
            num_heads (int): Number of heads in each attention layer
        """
        super().__init__(
            node_input_dim=node_input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        )
        self.node_f_dim = node_input_dim
        self.depot_f_dim = depot_input_dim
        self.emb_dim = embedding_dim

        self.depot_embed = nn.Linear(depot_input_dim, embedding_dim)

    def forward(self, x, depot_mask):
        """
        Calculates the node embedding for each node
        in each graph with regards to the depots.

        Args:
            x (torch.Tensor): Shape (num_graphs, num_nodes, num_features)
            depot_mask (torch.Tensor): Mask of the depot in each graph

        Returns:
            torch.Tensor: Returns the embedding of each node in each graph.
                Shape (num_graphs, num_nodes, embedding_dim).
        """
        batch_size, num_nodes, _ = x.size()

        not_depots = x[~depot_mask].view(batch_size, -1, self.node_f_dim)
        depots = x[depot_mask].view(batch_size, -1, self.node_f_dim)

        out = torch.cat(
            [
                self.depot_embed(depots[:, :, : self.depot_f_dim]),
                self.node_embed(not_depots[:, :, : self.node_f_dim]),
            ],
            dim=1,
        )

        # move embeddings to original matrix pos
        # depots shape: (batch, node_num, 1)
        # out shape: (batch, node_num, 128)
        temp = torch.empty_like(out)
        num_depot = len(torch.where(depot_mask[0] == 1))

        temp[depot_mask] = out[:, :num_depot, :].reshape(
            batch_size * num_depot, self.emb_dim
        )
        temp[~depot_mask] = out[:, num_depot:, :].reshape(
            (batch_size * (num_nodes - num_depot)), self.emb_dim
        )

        out = temp
        for layer in self.attention_layers:
            out = layer(out)

        return out


class BatchNorm(nn.Module):
    """
    Coverts inputs of (N, L, C) to (N*L, C)
    s.t we can apply BatchNorm for the
    features C.
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        shape = x.size()
        return self.norm(x.view(-1, shape[-1])).view(*shape)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int):
        """
        MultiHeadAttentionLayer with skip connection, batch
        normalization and a fully connected network.

        Args:
            embedding_dim (int): Number of dimensions in the embedding space.
            hidden_dim (int): Number of neurons of the hidden layer of the fcl.
            num_heads (int): Number of attention heads.
        """
        super().__init__()

        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )

        self.bn1 = BatchNorm(embedding_dim)
        self.bn2 = BatchNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape
                (num_graph, num_nodes, num_features)

        Returns:
            torch.Tensor: Output of shape
                (num_graph, num_nodes, embedding_dim)
        """
        out = self.bn1(x + self.attention_layer(x, x, x)[0])
        out = self.bn2(out + self.ff(out))

        return out
