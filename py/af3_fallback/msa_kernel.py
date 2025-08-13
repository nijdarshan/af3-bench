import torch


class _MSAWeightedAveragingFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, b, g):
        # v: (*, seq, res, heads, c_hidden)
        # b: (*, res, res, heads)
        # g: (*, seq, res, heads, c_hidden)
        weights = torch.softmax(b, dim=-2)
        weights = weights.unsqueeze(-4).unsqueeze(-1)  # (*, 1, res, res, heads, 1)
        v_exp = v.unsqueeze(-3)  # (*, seq, res, 1, heads, c_hidden)
        out = torch.sigmoid(g) * torch.sum(v_exp * weights, dim=-3)
        out = out.reshape(*out.shape[:-2], out.shape[-2] * out.shape[-1])
        return out


MSAWeightedAveragingFused = _MSAWeightedAveragingFused.apply

