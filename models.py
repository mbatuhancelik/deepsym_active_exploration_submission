import os
import time

import torch

import utils
from blocks import GumbelSigmoidLayer


class DeepSymbolGenerator:
    """DeepSym model from https://arxiv.org/abs/2012.02532"""

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module,
                 device: str, lr: float, path: str, coeff: float = 1.0, **kwargs):
        """
        Parameters
        ----------
        encoder : torch.nn.Module
            Encoder network.
        decoder : torch.nn.Module
            Decoder network.
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
        self.lr = lr

        self.optimizer = torch.optim.Adam(lr=lr, params=[
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()}])

        self.criterion = torch.nn.MSELoss()
        self.iteration = 0
        self.epoch = 0
        self.best_loss = 1e100
        self.path = path
        self.module_names = ["encoder", "decoder"]

    def encode(self, x: torch.Tensor, eval_mode=False) -> torch.Tensor:
        """
        Given a state, return its encoding

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
            # TODO: this should be different for gumbel softmax version
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
            The concatenation of the encoder's output and the action vector
            (i.e. the input of the decoder).
        """
        x = sample["state"]
        h = self.encode(x, eval_mode)
        a = sample["action"].to(self.device)
        z = torch.cat([h, a], dim=-1)
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
        for e in range(epoch):
            epoch_loss, time_elapsed = self.one_pass_optimize(loader)
            self.epoch += 1
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save("_best")
            print(f"epoch={self.epoch}, iter={self.iteration}, loss={epoch_loss:.5f}, elapsed={time_elapsed:.2f}")
            self.save("_last")

    def load(self, ext):
        for name in self.module_names:
            module_path = os.path.join(self.path, name+ext+".ckpt")
            module_dict = torch.load(module_path)
            getattr(self, name).load_state_dict(module_dict)

    def save(self, ext):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for name in self.module_names:
            module = getattr(self, name)
            module_dict = module.eval().cpu().state_dict()
            module_path = os.path.join(self.path, name+ext+".ckpt")
            torch.save(module_dict, module_path)
            getattr(self, name).train().to(self.device)

    def print_model(self, space=0, encoder_only=False):
        for name in self.module_names:
            utils.print_module(getattr(self, name), name, space)

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


class DeepSymv3(DeepSymbolGenerator):
    def __init__(self, **kwargs):
        super(DeepSymv3, self).__init__(**kwargs)
        # self.encoder_att = kwargs.get("encoder_att")
        self.decoder_att = kwargs.get("decoder_att")
        # self.discretization = GumbelSigmoidLayer(hard=False, T=1.0)
        # self.optimizer.param_groups.append(
        #         {"params": self.encoder_att.parameters(),
        #          "lr": self.lr,
        #          "betas": (0.9, 0.999),
        #          "eps": 1e-8,
        #          "amsgrad": False,
        #          "maximize": False,
        #          "weight_decay": 0})
        self.optimizer.param_groups.append(
                {"params": self.decoder_att.parameters(),
                 "lr": self.lr,
                 "betas": (0.9, 0.999),
                 "eps": 1e-8,
                 "amsgrad": False,
                 "maximize": False,
                 "weight_decay": 0})
        # self.module_names.append("encoder_att")
        self.module_names.append("decoder_att")

    def encode(self, x, pad_mask, eval_mode=False):
        n_sample, n_seg, ch, h, w = x.shape
        x = x.reshape(-1, ch, h, w)
        h = self.encoder(x.to(self.device))
        h = h.reshape(n_sample, n_seg, -1)
        # h_att, _ = self.encoder_att(h, h, h, key_padding_mask=~pad_mask.bool().to(self.device))
        # h_att = self.discretization(h_att)
        if eval_mode:
            # h_att = h_att.round()
            h = h.round()
        return h

    def concat(self, sample, eval_mode=False):
        x = sample["state"]
        h = self.encode(x, sample["pad_mask"], eval_mode)
        n_sample, n_seg = h.shape[0], h.shape[1]
        a = torch.tensor([[0., 1., 0.]], dtype=torch.float, device=self.device)
        a = a.repeat(n_sample, 1)
        a = a.unsqueeze(1).repeat(1, n_seg, 1)
        a[torch.arange(n_sample, device=self.device), sample["action"][:, 0].to(self.device)] = torch.tensor([1., 0., 0.], device=self.device)
        a[torch.arange(n_sample, device=self.device), sample["action"][:, 1].to(self.device)] = torch.tensor([0., 0., 1.], device=self.device)
        # a = sample["action"].to(self.device)
        # a = a.unsqueeze(1).repeat(1, h.shape[1], 1)
        z = torch.cat([h, a], dim=-1)
        z_att = self.decoder_att(z, src_key_padding_mask=~sample["pad_mask"].bool().to(self.device))
        return z_att

    def decode(self, z, mask):
        n_sample, n_seg, z_dim = z.shape
        z = z.reshape(-1, z_dim)
        e = self.decoder(z)
        e = e.reshape(n_sample, n_seg, -1)
        mask = mask.reshape(n_sample, n_seg, 1).to(self.device)
        # turn off computation for padded parts
        e_masked = e * mask
        return e_masked

    def forward(self, sample, eval_mode=False):
        z = self.concat(sample, eval_mode)
        e = self.decode(z, sample["pad_mask"])
        return z, e

    def loss(self, sample):
        e_truth = sample["effect"].to(self.device)
        _, e_pred = self.forward(sample)
        mask = sample["pad_mask"].to(self.device).unsqueeze(2)
        L = (((e_truth - e_pred) ** 2) * mask).sum(dim=[1, 2]).mean() * self.coeff
        return L


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
