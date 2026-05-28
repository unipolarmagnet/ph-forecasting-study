"""DLinear — verified implementation.

Source (verbatim architecture): cure-lab/LTSF-Linear, models/DLinear.py
Paper: Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023.
This is the implementation we reproduced against the paper's Table 2 (see archive/reproduction).
"""
import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of a time series."""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition into seasonal (residual) and trend (moving mean)."""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """Decomposition-Linear. Two linear maps (seasonal + trend) over the time axis.

    Args:
        seq_len: input window length
        pred_len: forecast horizon
        enc_in:  number of input channels (features)
        individual: separate linear layer per channel if True
        kernel_size: moving-average kernel for decomposition (default 25, as in paper)
    """
    def __init__(self, seq_len, pred_len, enc_in, individual=False, kernel_size=25,
                 mix_channels=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        self.mix_channels = mix_channels
        self.decompsition = series_decomp(kernel_size)

        if individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(enc_in)])
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(enc_in)])
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)

        # Channel-mixing head: combine the per-channel forecasts into the target
        # channel. This is what lets exogenous features (indicators, persistent-
        # homology columns) influence the target forecast — vanilla DLinear is
        # channel-independent and ignores them. Output becomes a single channel.
        if mix_channels:
            self.mix = nn.Linear(enc_in, 1)

    def forward(self, x):
        # x: [Batch, seq_len, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output          # [Batch, Channel, pred_len]
        x = x.permute(0, 2, 1)                       # [Batch, pred_len, Channel]
        if self.mix_channels:
            x = self.mix(x)                          # [Batch, pred_len, 1] (target)
        return x


def build(cfg):
    """Factory used by the shared harness. cfg is an argparse.Namespace."""
    return DLinear(
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        enc_in=cfg.enc_in,
        individual=getattr(cfg, 'individual', False),
        kernel_size=getattr(cfg, 'kernel_size', 25),
        mix_channels=getattr(cfg, 'mix_channels', False),
    )
