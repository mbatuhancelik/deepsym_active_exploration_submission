import os

import torch
import wandb

import utils
import blocks


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
        for i, sample in enumerate(loader):
            self.optimizer.zero_grad()
            L = self.loss(sample)
            L.backward()
            self.optimizer.step()
            avg_loss += L.item()
            self.iteration += 1
        avg_loss /= (i+1)
        return avg_loss

    def train(self, epoch, loader, val_loader=None):
        for e in range(epoch):
            # one epoch training over the train set
            epoch_loss = self.one_pass_optimize(loader)
            self.epoch += 1
            wandb.log({"train_loss": epoch_loss, "epoch": self.epoch})

            # calculate validation loss
            if val_loader is not None:
                self.eval_mode()
                val_loss = 0.0
                for i, sample in enumerate(val_loader):
                    with torch.no_grad():
                        L = self.loss(sample)
                    val_loss += L.item()
                val_loss /= (i+1)
                wandb.log({"val_loss": val_loss, "epoch": self.epoch})

                if val_loss < self.best_loss:
                    wandb.log({"best_val_loss": val_loss, "best_val_loss_epoch": self.epoch})
                    self.best_loss = val_loss
                    self.save("_best")
                print(f"epoch={self.epoch}, iter={self.iteration}, "
                      f"train loss={epoch_loss:.5f}, val loss={val_loss:.5f}")
                self.train_mode()
            else:
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save("_best")
                print(f"epoch={self.epoch}, iter={self.iteration}, loss={epoch_loss:.5f}")
            self.save("_last")
        self.save_wandb("_last")
        self.save_wandb("_best")

    def load(self, ext, from_wandb=False):
        for name in self.module_names:
            if from_wandb:
                module_path = os.path.join(self.path, name+ext+".pt")
                module_dict = wandb.restore(module_path, run_path=f"colorslab/multideepsym/{wandb.run.id}").name
                module_dict = torch.load(module_dict)
            else:
                module_path = os.path.join(self.path, name+ext+".pt")
                module_dict = torch.load(module_path)
            getattr(self, name).load_state_dict(module_dict)

    def save(self, ext):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for name in self.module_names:
            module = getattr(self, name)
            module_dict = module.eval().cpu().state_dict()
            module_path = os.path.join(self.path, name+ext+".pt")
            torch.save(module_dict, module_path)
            getattr(self, name).train().to(self.device)

    def save_wandb(self, ext):
        wandb.save(os.path.join(self.path, "*"+ext+".pt"))

    def print_model(self, space=0):
        for name in self.module_names:
            utils.print_module(getattr(self, name), name, space)

    def eval_mode(self):
        for name in self.module_names:
            module = getattr(self, name)
            module.eval()

    def train_mode(self):
        for name in self.module_names:
            module = getattr(self, name)
            module.train()


class MultiDeepSym(DeepSymbolGenerator):
    def __init__(self, **kwargs):
        super(MultiDeepSym, self).__init__(**kwargs)
        self._append_module("feedforward", kwargs.get("feedforward"))
        self._append_module("attention", kwargs.get("attention"))
        self._append_module("pre_attention_mlp", kwargs.get("pre_attention_mlp"))

    def _append_module(self, name, module):
        setattr(self, name, module)
        self.module_names.append(name)
        self.optimizer.add_param_group({"params": module.parameters()})

    def encode(self, x, eval_mode=False):
        n_sample, n_seg, ch, h, w = x.shape
        x = x.reshape(-1, ch, h, w)
        h = self.encoder(x.to(self.device))
        h = h.reshape(n_sample, n_seg, -1)
        if eval_mode:
            h = h.round()
        return h

    def attn_weights(self, x, pad_mask, eval_mode=False):
        # assume that x is not an image for the moment..
        n_sample, n_seg, n_feat = x.shape
        x = x.reshape(-1, n_feat)
        x = self.pre_attention_mlp(x.to(self.device))
        x = x.reshape(n_sample, n_seg, -1)
        attn_weights = self.attention(x, src_key_mask=pad_mask.to(self.device))
        if eval_mode:
            attn_weights = attn_weights.round()
        return attn_weights

    def concat(self, sample, eval_mode=False):
        x = sample["state"]
        a = sample["action"].to(self.device)
        h = self.encode(x, eval_mode)
        z = torch.cat([h, a], dim=-1)
        return z

    def aggregate(self, z, attn_weights):
        n_batch, n_seg, n_dim = z.shape
        h = self.feedforward(z.reshape(-1, n_dim)).reshape(n_batch, n_seg, -1).unsqueeze(1)
        att_out = attn_weights @ h  # (n_batch, n_head, n_seg, n_dim)
        att_out = att_out.permute(0, 2, 1, 3).reshape(n_batch, n_seg, -1)  # (n_batch, n_seg, n_head*n_dim)
        return att_out

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
        attn_weights = self.attn_weights(sample["state"], sample["pad_mask"], eval_mode)
        z_att = self.aggregate(z, attn_weights)
        e = self.decode(z_att, sample["pad_mask"])
        return z, attn_weights, e

    def loss(self, sample):
        e_truth = sample["effect"].to(self.device)
        _, _, e_pred = self.forward(sample)
        mask = sample["pad_mask"].to(self.device).unsqueeze(2)
        L = (((e_truth - e_pred) ** 2) * mask).sum(dim=[1, 2]).mean() * self.coeff
        return L

    def loss_with_pred(self, sample):
        e_truth = sample["effect"].to(self.device)
        _, _, e_pred = self.forward(sample)
        mask = sample["pad_mask"].to(self.device).unsqueeze(2)
        L = (((e_truth - e_pred) ** 2) * mask).sum(dim=[1, 2]).mean() * self.coeff
        return L, e_pred


class MultiDeepSymMLP(MultiDeepSym):
    def __init__(self, **kwargs):
        super(MultiDeepSymMLP, self).__init__(**kwargs)

    def encode(self, x, eval_mode=False):
        n_sample, n_seg, n_feat = x.shape
        x = x.reshape(-1, n_feat)
        h = self.encoder(x.to(self.device))
        h = h.reshape(n_sample, n_seg, -1)
        if eval_mode:
            h = h.round()
        return h


class SymbolForward(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        enc_layers = [torch.nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            enc_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.encoder = torch.nn.ModuleList(enc_layers)

        obj_dec_layers = [num_heads*hidden_dim] + [hidden_dim] * (num_layers - 2) + \
                         [output_dim]
        self.obj_decoder = torch.nn.Sequential(
            blocks.MLP(obj_dec_layers),
            torch.nn.Tanh())

        rel_dec_layers = [num_heads*hidden_dim] + [hidden_dim] * (num_layers - 2) + \
                         [num_heads*hidden_dim]
        self.rel_decoder = torch.nn.Sequential(
            blocks.MLP(rel_dec_layers),
            torch.nn.Tanh())

    def forward(self, x, attn_weights):
        n_batch, n_token, _ = x.shape
        x = x.unsqueeze(1)
        for layer in self.encoder:
            x = torch.nn.functional.relu(layer(x))
            x = attn_weights @ x
        # (batch, head, token, dim) -> (batch, token, head*dim)
        x = x.permute(0, 2, 1, 3).reshape(n_batch, n_token, -1)
        obj_pred = self.obj_decoder(x)
        rel_pred = self.rel_decoder(x).reshape(n_batch, n_token, self.num_heads, -1)
        rel_pred = rel_pred.permute(0, 2, 1, 3)
        rel_pred = (rel_pred @ rel_pred.permute(0, 1, 3, 2)) / (rel_pred.shape[-1]**0.5)
        return obj_pred, rel_pred
