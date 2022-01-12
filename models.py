import torch


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
