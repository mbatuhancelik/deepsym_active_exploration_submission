import os
import time

import torch

import utils


class DeepSymbolGenerator:
    """DeepSym model from https://arxiv.org/abs/2012.02532"""

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module,
                 subnetworks: list, device: str, lr: float,
                 path: str, coeff: float = 1.0):
        """
        Parameters
        ----------
        encoder : torch.nn.Module
            Encoder network.
        decoder : torch.nn.Module
            Decoder network.
        subnetworks : list of torch.nn.Module
            Optional list of subnetworks to use their output as input
            for the decoder.
        device : str
            The device of the networks.
        lr : float
            Learning rate.
        path : str
            Save and load path.
        coeff : float
            A hyperparameter to increase to speed of convergence when there
            are lots of zero values in the effect prediction (e.g. tile puzzle).
        """
        self.device = device
        self.coeff = coeff
        self.encoder = encoder
        self.decoder = decoder
        self.subnetworks = subnetworks

        self.optimizer = torch.optim.Adam(lr=lr, params=[
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()}])

        self.criterion = torch.nn.MSELoss()
        self.iteration = 0
        self.path = path

    def encode(self, x: torch.Tensor, eval_mode=False) -> torch.Tensor:
        """
        Given a state, return its encoding with the current
        encoder (i.e. no subnetwork code).

        Parameters
        ----------
        x : torch.Tensor
            The state tensor.

        Returns
        -------
        h : torch.Tensor
            The code of the given state.
        """
        h = self.encoder(x.to(self.device))
        if eval_mode:
            h = h.round()
        return h

    def concat(self, sample: dict, eval_mode=False) -> torch.Tensor:
        """
        Given a sample, return the concatenation of the encoders'
        output and the action vector.

        Parameters
        ----------
        sample : dict
            The input dictionary. This dict should containt following
            keys: `state` and `action`.

        Returns
        -------
        z : torch.Tensor
            The concatenation of the encoder's output, subnetworks'
            encoders output, and the action vector (i.e. the input
            of the decoder).
        """
        h = []
        x = sample["state"]
        h.append(self.encode(x, eval_mode))
        for network in self.subnetworks:
            with torch.no_grad():
                h.append(network.encode(x, eval_mode))
        h.append(sample["action"].to(self.device))
        z = torch.cat(h, dim=-1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Given a code, return the effect.

        Parameters
        ----------
        z : torch.Tensor
            The code tensor.

        Returns
        -------
        e : torch.Tensor
            The effect tensor.
        """
        e = self.decoder(z)
        return e

    def forward(self, sample, eval_mode=False):
        z = self.concat(sample, eval_mode)
        e = self.decode(z)
        return z, e

    def loss(self, sample):
        e_truth = sample["effect"].to(self.device)
        _, e_pred = self.forward(sample)
        L = self.criterion(e_pred, e_truth)*self.coeff
        return L

    def one_pass_optimize(self, loader):
        avg_loss = 0.0
        start = time.time()
        for i, sample in enumerate(loader):
            self.optimizer.zero_grad()
            L = self.loss(sample)
            L.backward()
            self.optimizer.step()
            avg_loss += L.item()
            self.iteration += 1
        end = time.time()
        avg_loss /= (i+1)
        time_elapsed = end-start
        return avg_loss, time_elapsed

    def train(self, epoch, loader):
        best_loss = 1e100
        for e in range(epoch):
            epoch_loss, time_elapsed = self.one_pass_optimize(loader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save("_best")
            print(f"epoch={e+1}, iter={self.iteration}, loss={epoch_loss:.5f}, elapsed={time_elapsed:.2f}")
            self.save("_last")

    def load(self, ext):
        encoder_path = os.path.join(self.path, "encoder"+ext+".ckpt")
        decoder_path = os.path.join(self.path, "decoder"+ext+".ckpt")
        encoder_dict = torch.load(encoder_path)
        decoder_dict = torch.load(decoder_path)
        self.encoder.load_state_dict(encoder_dict)
        self.decoder.load_state_dict(decoder_dict)

    def save(self, ext):
        encoder_dict = self.encoder.eval().cpu().state_dict()
        decoder_dict = self.decoder.eval().cpu().state_dict()
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        encoder_path = os.path.join(self.path, "encoder"+ext+".ckpt")
        decoder_path = os.path.join(self.path, "decoder"+ext+".ckpt")
        torch.save(encoder_dict, encoder_path)
        torch.save(decoder_dict, decoder_path)
        self.encoder.train().to(self.device)
        self.decoder.train().to(self.device)

    def print_model(self, space=0, encoder_only=False):
        utils.print_module(self.encoder, "Encoder", space)
        if not encoder_only:
            utils.print_module(self.decoder, "Decoder", space)
        if len(self.subnetworks) != 0:
            print("-"*15)
            print("  Subnetworks  ")
            print("-"*15)

        tab_length = 4
        for i, network in enumerate(self.subnetworks):
            print(" "*tab_length+"%d:" % (i+1))
            network.print_model(space=space+tab_length, encoder_only=True)
            print()

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()

    def train_mode(self):
        self.encoder.train()
        self.decoder.train()


class DeepSymv2(DeepSymbolGenerator):
    def __init__(self, **kwargs):
        super(DeepSymv2, self).__init__(**kwargs)

    def forward(self, sample, eval_mode=False):
        x = sample["state"]
        xn = sample["state_n"]
        bs = x.shape[0]
        x_all = torch.cat([x, xn], dim=0)
        z_all = self.encode(x_all, eval_mode=eval_mode)
        z_diff = z_all[bs:] - z_all[:bs]
        e = self.decode(z_diff)
        return z_diff, e

    def concat(self, **kwargs):
        raise NotImplementedError


class RBM(torch.nn.Module):

    def __init__(self, v_dim, h_dim):
        super(RBM, self).__init__()
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.w = torch.nn.Parameter(torch.nn.init.xavier_normal(torch.empty(v_dim, h_dim)))
        self.a = torch.nn.Parameter(torch.zeros(v_dim))
        self.b = torch.nn.Parameter(torch.zeros(h_dim))

    def sample_h(self, v):
        prob = torch.sigmoid(v @ self.w + self.b)
        with torch.no_grad():
            sample = prob.bernoulli()
        return prob, sample

    def sample_v(self, h):
        prob = torch.sigmoid(h @ self.w.t() + self.a)
        with torch.no_grad():
            sample = prob.bernoulli()
        return prob, sample

    def energy(self, v, h):
        return -(v @ self.a + h @ self.b + ((v @ self.w)*h).sum(dim=-1)) 

    def gibbs_k(self, h, k, prob=False):
        for _ in range(k):
            v_p, v = self.sample_v(h)
            h_p, h = self.sample_h(v)
        if prob:
            return v_p, h_p
        else:
            return v, h

    def forward(self, v):
        return self.sample_h(v)

    def extra_repr(self):
        return "v_dim={}, h_dim={}".format(self.v_dim, self.h_dim)
