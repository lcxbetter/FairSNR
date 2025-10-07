import torch
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn as nn
from torch_geometric.nn import GINConv, SAGEConv, GCNConv
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
class MLP_classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_classifier, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, args.num_classes)

    def clip_parameters(self):
        for p in self.lin.parameters():
            p.data.clamp_(-self.args.clip_c, self.args.clip_c)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None):
        h = self.lin(h)

        return h


class MLP_encoder(torch.nn.Module):
    def __init__(self, args):
        super(MLP_encoder, self).__init__()
        self.args = args

        self.lin = Linear(args.num_features, args.hidden)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index=None, mask_node=None):
        h = self.lin(x)

        return h   

class Intervenor_encoder(torch.nn.Module):
    def __init__(self, args):
        super(Intervenor_encoder, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, args.hidden)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index=None, mask_node=None):
        z = self.lin(x)
        return z   



class GCN_encoder(nn.Module):
    def __init__(self, args):
        super(GCN_encoder, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hidden)
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Dropout(p=args.dropout)
        )
        self.conv2 = GCNConv(args.hidden, args.hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.transition(x)
        h = self.conv2(x, edge_index, edge_weight)
        return h

class VGAE(nn.Module):
    def __init__(self, args):
        super(VGAE, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hidden)
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Dropout(p=args.dropout)
        )
        self.conv2_mean = GCNConv(args.hidden, args.hidden)
        self.conv2_var = GCNConv(args.hidden, args.hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2_mean.reset_parameters()
        self.conv2_var.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.transition(x)
        m = self.conv2_mean(x, edge_index, edge_weight)
        v = F.softplus(self.conv2_var(x, edge_index, edge_weight))
        return m, v
    def kl_loss(self,mu, log_std):
        return -0.5 * (1 + 2 * log_std - mu.pow(2) - log_std.exp().pow(2)).mean()



class GIN_encoder(nn.Module):
    def __init__(self, args):
        super(GIN_encoder, self).__init__()

        self.args = args

        self.mlp = nn.Sequential(
            nn.Linear(args.num_features, args.hidden),
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Linear(args.hidden, args.hidden),
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden)
        )

        self.conv = GINConv(self.mlp)

    def reset_parameters(self):
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, adj_norm_sp=None):
        h = self.conv(x, edge_index)
        return h

class SAGE_encoder(nn.Module):
    def __init__(self, args):
        super(SAGE_encoder, self).__init__()

        self.args = args

        self.conv1 = SAGEConv(args.num_features, args.hidden, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Dropout(p=args.dropout)
        )
        self.conv2 = SAGEConv(args.hidden, args.hidden, normalize=True)
        self.conv2.aggr = 'mean'

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.transition(x)
        h = self.conv2(x, edge_index, edge_weight)
        return h



class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 18),
            nn.ReLU(),
            nn.Linear(18, 1)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.layer(x)

    def reset_parameters(self):
        for layer in self.layer:
            if isinstance(layer, nn.Linear):
                # 改为He初始化
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                init.constant_(layer.bias, 0)
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)