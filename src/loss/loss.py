import torch
from torch import nn


class SDRiLoss(nn.Module):
    # [see] 3.4 Multi-loss
    def __init__(self, alpha=0.4, tau=30.0):
        super().__init__()
        self.initial_alpha = alpha
        self.alpha = alpha
        self.tau = tau

    def set_epoch(self, epoch):
        decays = max(0, epoch - 100) // 5
        self.alpha = self.initial_alpha * 0.8 ** decays

    def pit_si_snr(self, estimates, targets, lengths):
        count = lengths[:, None, None]
        time = torch.arange(estimates.size(-1), device=estimates.device)
        mask = time[None, None, :] < count

        def center(x):
            x = x.float()
            mean = (x * mask).sum(dim=-1, keepdim=True) / count
            return (x - mean) * mask

        estimates, targets = center(estimates), center(targets)

        def score(ref):
            dot = (estimates * ref).sum(dim=-1, keepdim=True)
            energy = ref.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
            projection = dot / energy * ref
            signal = projection.square().sum(dim=-1).clamp_min(1e-8)
            noise = (estimates - projection).square().sum(dim=-1).clamp_min(1e-8)
            si_snr = 10 * (signal.log10() - noise.log10())
            return si_snr.clamp(max=self.tau).sum(dim=1)

        return -torch.maximum(score(targets), score(targets.flip(1))).mean()

    def forward(self, logits, targets, lengths, aux_logits=(), **batch):
        main = self.pit_si_snr(logits, targets, lengths)
        if not aux_logits:
            return {"loss": main}

        auxiliary = torch.stack([self.pit_si_snr(estimate, targets, lengths) for estimate in aux_logits]).mean()
        return {"loss": (1 - self.alpha) * main + self.alpha * auxiliary}