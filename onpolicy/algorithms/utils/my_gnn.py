import torch
import torch.nn as nn
import torch.nn.functional as F

class MyGNN(nn.Module):
    def __init__(self, layers, node_dim, edge_dim, hidden_dim, output_dim, dropout_rate=0.0,jk=False, activation="relu", aggregation="sum",**kwargs):
        super(MyGNN, self).__init__()

        self.layers = layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation_name = activation
        self.aggregation = aggregation
        self.jk = jk  # whether to use Jumping Knowledge (JK) mechanism

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # GNN layers
    
        self.gnn_layers = nn.ModuleList()

        for k in range(layers):
            if k == 0:
                input_node_dim = node_dim
            else:
                input_node_dim = hidden_dim
                
            message_dim = input_node_dim + edge_dim
            update_input_dim = input_node_dim + message_dim

            if k == layers - 1:
                layer_output_dim = output_dim
            else:
                layer_output_dim = hidden_dim

            self.gnn_layers.append(nn.Linear(update_input_dim, layer_output_dim))
            
            if self.jk:
                self.jk_linear = nn.Linear(layers * hidden_dim, output_dim)
            else:
                self.jk_linear = None
            
        self.dropout = nn.Dropout(dropout_rate)
            

    def forward(self, x, edge_attr, edge_index):

        h = x  # h^0_v <- x_v
        
        if self.jk:
            layer_outputs = []

        for k, layer in enumerate(self.gnn_layers):

            source = edge_index[0]  # source nodes u
            target = edge_index[1]  # destination nodes v

            source_features = h[source]  
            messages = torch.cat([source_features, edge_attr], dim=-1)  # m_{u -> v}^k <- CONCAT(h_u^{k-1}, x_{u,v})

            # h_N(v)^k <- AGGREGATE({m_{u,v}^k})
            
            aggregated = torch.zeros(h.shape[0], messages.shape[-1], device=h.device, dtype=h.dtype)

            if self.aggregation == "sum":

                aggregated.index_add_(0, target, messages)

            elif self.aggregation == "mean":

                aggregated.index_add_(0,target, messages)

                degree = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
                degree.index_add_(0, target,torch.ones(target.shape[0], device=h.device, dtype=h.dtype))
                degree = degree.clamp(min=1.0).unsqueeze(-1)

                aggregated = aggregated/degree

            else:
                raise ValueError(f"Unsupported aggregation: {self.aggregation}")


            # h_v^k <- sigma(W_k * CONCAT(h_v^{k-1}, h_N(v)^k))
            
            update_input = torch.cat([h, aggregated],dim=-1)
            h = layer(update_input)
            h = self.activation(h)
            
            if k < len(self.gnn_layers)-1:   # dropout layer
                h = self.dropout(h)

            # h_v^k <- h_v^k / ||h_v^k||_2  (L2 regularization)
            
            #h = F.normalize(h,p=2,dim=-1)
            
            if self.jk:
                layer_outputs.append(h)
            
        if self.jk:
            
            # Jumping Knowledge
            h_jk = torch.cat(layer_outputs, dim=-1)
            h = self.jk_linear(h_jk)
    
        z = h  # z_v <- h_v^K

        return z