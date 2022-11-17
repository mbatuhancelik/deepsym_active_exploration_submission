import math
import os
import torch
import numpy


class StraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.clamp(-1., 1.)


class STLayer(torch.nn.Module):

    def __init__(self):
        super(STLayer, self).__init__()
        self.func = StraightThrough.apply

    def forward(self, x):
        return self.func(x)


class STSigmoid(torch.nn.Module):
    def __init__(self):
        super(STSigmoid, self).__init__()

    def forward(self, x):
        m = torch.distributions.Bernoulli(logits=x)
        sample = m.sample()
        probs = torch.sigmoid(x)
        sample = sample + probs - probs.detach()
        return sample


class GumbelSigmoidLayer(torch.nn.Module):
    def __init__(self, hard=False, T=1.0):
        super(GumbelSigmoidLayer, self).__init__()
        self.hard = hard
        self.T = T

    def forward(self, x):
        return gumbel_sigmoid(x, self.T, self.hard)


class GumbelSoftmaxLayer(torch.nn.Module):
    def __init__(self, hard=False, T=1.0):
        super(GumbelSoftmaxLayer, self).__init__()
        self.hard = hard
        self.T = T

    def forward(self, x):
        return torch.nn.functional.gumbel_softmax(x, tau=self.T, hard=self.hard)


class Linear(torch.nn.Module):
    """ linear layer with optional batch normalization. """
    def __init__(self, in_features, out_features, std=None, batch_norm=False, gain=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(num_features=self.out_features)

        if std is not None:
            self.weight.data.normal_(0., std)
            self.bias.data.normal_(0., std)
        else:
            # defaults to linear activation
            if gain is None:
                gain = 1
            stdv = math.sqrt(gain / self.weight.size(1))
            self.weight.data.normal_(0., stdv)
            self.bias.data.zero_()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        if hasattr(self, "batch_norm"):
            x = self.batch_norm(x)
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}".format(self.in_features, self.out_features)


class MLP(torch.nn.Module):
    """ multi-layer perceptron with batch norm option """
    def __init__(self, layer_info, activation=torch.nn.ReLU(), std=None, batch_norm=False, indrop=None, hiddrop=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_info[0]
        for i, unit in enumerate(layer_info[1:-1]):
            if i == 0 and indrop:
                layers.append(torch.nn.Dropout(indrop))
            elif i > 0 and hiddrop:
                layers.append(torch.nn.Dropout(hiddrop))
            layers.append(Linear(in_features=in_dim, out_features=unit, std=std, batch_norm=batch_norm, gain=2))
            layers.append(activation)
            in_dim = unit
        layers.append(Linear(in_features=in_dim, out_features=layer_info[-1], batch_norm=False))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def load(self, path, name):
        state_dict = torch.load(os.path.join(path, name+".ckpt"))
        self.load_state_dict(state_dict)

    def save(self, path, name):
        dv = self.layers[-1].weight.device
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.cpu().state_dict(), os.path.join(path, name+".ckpt"))
        self.train().to(dv)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        self.block = [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias)]
        if batch_norm:
            self.block.append(torch.nn.BatchNorm2d(out_channels))
        self.block.append(torch.nn.ReLU())

        if std is not None:
            self.block[0].weight.data.normal_(0., std)
            self.block[0].bias.data.normal_(0., std)
        self.block = torch.nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class ConvTransposeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True,
                 batch_norm=False):
        super(ConvTransposeBlock, self).__init__()
        self.block = [torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if batch_norm:
            self.block.append(torch.nn.BatchNorm2d(out_channels))
        self.block.append(torch.nn.ReLU())

        if std is not None:
            self.block[0].weight.data.normal_(0., std)
            self.block[0].bias.data.normal_(0., std)
        self.block = torch.nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class HME(torch.nn.Module):

    def __init__(self, gating_features, in_features, out_features, depth, projection='linear', gumbel="none"):
        super(HME, self).__init__()
        self.gating_features = gating_features
        self.in_features = in_features
        self.out_features = out_features
        self.depth = depth
        self.proj = projection
        self.gumbel = gumbel
        self.n_leaf = int(2**depth)
        self.gate_count = int(self.n_leaf - 1)
        self.gw = torch.nn.Parameter(
            torch.nn.init.kaiming_normal_(
                torch.empty(self.gate_count, gating_features), nonlinearity='sigmoid').t())
        self.gb = torch.nn.Parameter(torch.zeros(self.gate_count))
        if self.proj == 'linear':
            self.pw = torch.nn.init.kaiming_normal_(torch.empty(out_features*self.n_leaf, in_features),
                                                    nonlinearity='linear')
            self.pw = torch.nn.Parameter(self.pw.reshape(out_features, self.n_leaf, in_features).permute(0, 2, 1))
            self.pb = torch.nn.Parameter(torch.zeros(out_features, self.n_leaf))
        elif self.proj == 'constant':
            self.z = torch.nn.Parameter(torch.randn(out_features, self.n_leaf))

    def forward(self, x_gating, x_leaf):
        node_densities = self.node_densities(x_gating)
        leaf_probs = node_densities[:, -self.n_leaf:].t()

        if self.proj == 'linear':
            gated_projection = (self.pw @ leaf_probs).permute(2, 0, 1)
            gated_bias = (self.pb @ leaf_probs).permute(1, 0)
            result = (gated_projection @ x_leaf.reshape(-1, self.in_features, 1))[:, :, 0] + gated_bias
        elif self.proj == 'constant':
            result = (self.z @ leaf_probs).permute(1, 0)

        return result

    def node_densities(self, x):
        gatings = self.gatings(x)
        node_densities = torch.ones(x.shape[0], 2**(self.depth+1)-1, device=x.device)
        it = 1
        for d in range(1, self.depth+1):
            for i in range(2**d):
                parent_index = (it+1) // 2 - 1
                child_way = (it+1) % 2
                if child_way == 0:
                    parent_gating = gatings[:, parent_index]
                else:
                    parent_gating = 1 - gatings[:, parent_index]
                parent_density = node_densities[:, parent_index].clone()
                node_densities[:, it] = (parent_density * parent_gating)
                it += 1
        return node_densities

    def gatings(self, x):
        if self.gumbel == "none":
            g = torch.sigmoid(x @ self.gw + self.gb)
        else:
            hard = True if self.gumbel == "hard" else False
            g = gumbel_sigmoid(x @ self.gw + self.gb, T=1.0, hard=hard)
        return g

    def total_path_value(self, z, index, level=None):
        gatings = self.gatings(z)
        gateways = numpy.binary_repr(index, width=self.depth)
        L = 0.
        current = 0
        if level is None:
            level = self.depth

        for i in range(level):
            if int(gateways[i]) == 0:
                L += gatings[:, current].mean()
                current = 2 * current + 1
            else:
                L += (1 - gatings[:, current]).mean()
                current = 2 * current + 2
        return L

    def extra_repr(self):
        return "gating_features=%d, in_features=%d, out_features=%d, depth=%d, projection=%s" % (
            self.gating_features,
            self.in_features,
            self.out_features,
            self.depth,
            self.proj)


class Flatten(torch.nn.Module):
    def __init__(self, dims):
        super(Flatten, self).__init__()
        self.dims = dims

    def forward(self, x):
        dim = 1
        for d in self.dims:
            dim *= x.shape[d]
        return x.reshape(-1, dim)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        o = x.reshape(self.shape)
        return o


class Avg(torch.nn.Module):
    def __init__(self, dims):
        super(Avg, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.mean(dim=self.dims)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"


class ChannelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ChannelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B*C, 1, H, W)
        h = self.model(x)
        h = h.reshape(B, -1)
        return h


def sample_gumbel_diff(*shape):
    eps = 1e-20
    u1 = torch.rand(shape)
    u2 = torch.rand(shape)
    diff = torch.log(torch.log(u2+eps)/torch.log(u1+eps)+eps)
    return diff


def gumbel_sigmoid(logits, T=1.0, hard=False):
    g = sample_gumbel_diff(*logits.shape)
    g = g.to(logits.device)
    y = (g + logits) / T
    s = torch.sigmoid(y)
    if hard:
        s_hard = s.round()
        s = (s_hard - s).detach() + s
    return s
