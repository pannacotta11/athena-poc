import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        """
        Reversible Instance Normalization
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        # initialize affine parameters
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = 1
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
            x = x * self.stdev + self.mean
        return x

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, input_dim=1):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Use RevIN with affine=True for better accuracy
        self.revin = RevIN(input_dim, affine=True)
        self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        x = self.revin(x, 'norm')
        x = x.permute(0, 2, 1)
        x = self.Linear(x)
        x = x.permute(0, 2, 1)
        x = self.revin(x, 'denorm')
        return x