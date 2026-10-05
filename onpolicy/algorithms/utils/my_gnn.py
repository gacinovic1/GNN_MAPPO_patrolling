import torch
import torch.nn as nn


class MyGNN(nn.Module):

    def __init__(self,layers,node_dim,edge_dim,hidden_dim,output_dim, node_type_idx,node_type_dim=1,node_type_embed_dim=2,node_embedding_num=2,dropout_rate=0.0,jk=False, activation="relu", aggregation="sum"):
        super(MyGNN, self).__init__()

    def forward(self, x, edge_attr,edge_index):
        z = None
        return z