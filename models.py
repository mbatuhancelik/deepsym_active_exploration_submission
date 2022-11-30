import os
import time

import torch

import utils


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

    #calculates loss of sample
    def loss(self, sample):
        e_truth = sample["effect"].to(self.device)
        _, e_pred = self.forward(sample)
        L = self.criterion(e_pred, e_truth)*self.coeff
        return L

    #calculates cost function value
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

    def train(self, epoch, loader, val_loader=None):
        for e in range(epoch):
            # one epoch training over the train set
            epoch_loss, time_elapsed = self.one_pass_optimize(loader)
            self.epoch += 1

            # calculate validation loss
            if val_loader is not None:
                val_loss = 0.0
                for i, sample in enumerate(val_loader):
                    with torch.no_grad():
                        L = self.loss(sample)
                    val_loss += L.item()
                val_loss /= (i+1)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save("_best")
                print(f"epoch={self.epoch}, iter={self.iteration}, train loss={epoch_loss:.5f}, val loss={val_loss:.5f}, elapsed={time_elapsed:.2f}")
            else:
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


class DeepSymSegmentor(DeepSymbolGenerator):
    def __init__(self, **kwargs):
        super(DeepSymSegmentor, self).__init__(**kwargs)
        self.projector = kwargs.get("projector")
        self.decoder_att = kwargs.get("decoder_att")
        self.optimizer.param_groups.append(
                {"params": self.projector.parameters(),
                 "lr": self.lr,
                 "betas": (0.9, 0.999),
                 "eps": 1e-8,
                 "amsgrad": False,
                 "maximize": False,
                 "weight_decay": 0})
        self.optimizer.param_groups.append(
                {"params": self.decoder_att.parameters(),
                 "lr": self.lr,
                 "betas": (0.9, 0.999),
                 "eps": 1e-8,
                 "amsgrad": False,
                 "maximize": False,
                 "weight_decay": 0})
        self.module_names.append("projector")
        self.module_names.append("decoder_att")

        self.max_parts = kwargs.get("max_parts")
        self.part_dims = kwargs.get("part_dims")

    def encode(self, x, eval_mode=False):
        h_aug = self.encoder(x.to(self.device))
        gating, h = h_aug[:, :self.max_parts], h_aug[:, self.max_parts:]
        n_sample = h.shape[0]
        h = h.reshape(n_sample, self.max_parts, self.part_dims)
        gating = gating.unsqueeze(2)
        h_filtered = gating * h
        return h_filtered, gating

    def concat(self, sample, eval_mode=False):
        x = sample["state"]
        a = sample["action"].to(self.device)
        h, g = self.encode(x, eval_mode)
        a = a.unsqueeze(1)
        a = a.repeat(1, self.max_parts, 1)
        z = torch.cat([h, a], dim=-1)
        return z, g

    def aggregate(self, z, g):
        z = self.projector(z)
        z_att = self.decoder_att(z, src_key_padding_mask=~(g[..., 0].round().bool()))
        return z_att

    def decode(self, z, g):
        n_sample, n_part, z_dim = z.shape
        z = z.reshape(-1, z_dim)
        e = self.decoder(z)
        _, ch, h, w = e.shape
        e = e.reshape(n_sample, n_part, -1)
        e_gated = (e * g).sum(dim=1)
        e_gated = e_gated.reshape(n_sample, ch, h, w)
        return e_gated

    def forward(self, sample, eval_mode=False):
        z, g = self.concat(sample, eval_mode)
        z_att = self.aggregate(z, g)
        e = self.decode(z_att, g)
        return e, z, g

    def loss(self, sample):
        e_truth = sample["effect"].to(self.device)
        e_pred, _, _ = self.forward(sample)
        L = self.criterion(e_pred, e_truth) * self.coeff
        return L


class DeepSymv3(DeepSymbolGenerator):
    def __init__(self, **kwargs):
        super(DeepSymv3, self).__init__(**kwargs)
        self.projector = kwargs.get("projector")
        self.decoder_att = kwargs.get("decoder_att")
        self.optimizer.param_groups.append(
                {"params": self.projector.parameters(),
                 "lr": self.lr,
                 "betas": (0.9, 0.999),
                 "eps": 1e-8,
                 "amsgrad": False,
                 "maximize": False,
                 "weight_decay": 0})
        self.optimizer.param_groups.append(
                {"params": self.decoder_att.parameters(),
                 "lr": self.lr,
                 "betas": (0.9, 0.999),
                 "eps": 1e-8,
                 "amsgrad": False,
                 "maximize": False,
                 "weight_decay": 0})
        self.module_names.append("projector")
        self.module_names.append("decoder_att")

    def encode(self, x, pad_mask, eval_mode=False):
        n_sample, n_seg, ch, h, w = x.shape
        x = x.reshape(-1, ch, h, w)
        h = self.encoder(x.to(self.device))
        h = h.reshape(n_sample, n_seg, -1)
        if eval_mode:
            h = h.round()
        return h

    def concat(self, sample, eval_mode=False):
        x = sample["state"]
        a = sample["action"].to(self.device)
        h = self.encode(x, sample["pad_mask"], eval_mode)
        z = torch.cat([h, a], dim=-1)
        return z

    def aggregate(self, z, pad_mask):
        z = self.projector(z)
        z_att = self.decoder_att(z, src_key_padding_mask=~pad_mask.bool().to(self.device))
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
        z_att = self.aggregate(z, sample["pad_mask"])
        e = self.decode(z_att, sample["pad_mask"])
        return z, e

    def loss(self, sample):
        e_truth = sample["effect"].to(self.device)
        _, e_pred = self.forward(sample)
        mask = sample["pad_mask"].to(self.device).unsqueeze(2)
        L = (((e_truth - e_pred) ** 2) * mask).sum(dim=[1, 2]).mean() * self.coeff
        return L


class DeepSymv4(DeepSymbolGenerator):
    def __init__(self, **kwargs):
        super(DeepSymv4, self).__init__(**kwargs)
        self.decoder_att = kwargs.get("decoder_att")
        self.optimizer.param_groups.append(
                {"params": self.decoder_att.parameters(),
                 "lr": self.lr,
                 "betas": (0.9, 0.999),
                 "eps": 1e-8,
                 "amsgrad": False,
                 "maximize": False,
                 "weight_decay": 0})
        self.module_names.append("decoder_att")

    def encode(self, x, pad_mask, eval_mode=False):
        h = self.encoder(x.to(self.device))
        if eval_mode:
            h = h.round()
        return h

    def concat(self, sample, eval_mode=False):
        x = sample["state"]
        h = self.encode(x, sample["pad_mask"], eval_mode)
        n_sample, n_seg = h.shape[0], h.shape[1]
        a = sample["action"].repeat_interleave(n_seg, 0).reshape(n_sample, n_seg, 17).to(self.device)
        z = torch.cat([h, a], dim=-1)
        return z

    def aggregate(self, z, pad_mask):
        z_att = self.decoder_att(z, src_key_padding_mask=~pad_mask.bool().to(self.device))
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
        z_att = self.aggregate(z, sample["pad_mask"])
        e = self.decode(z_att, sample["pad_mask"])
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
