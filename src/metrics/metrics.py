import torch

from src.metrics.base_metric import BaseMetric

def si_snr(est, ref, eps=1e-8):
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    s_target = (est * ref).sum(dim=-1, keepdim=True) / ((ref * ref).sum(dim=-1, keepdim=True) + eps) * ref
    e_noise = est - s_target
    return 10 * torch.log10(((s_target * s_target).sum(dim=-1) + eps) / ((e_noise * e_noise).sum(dim=-1) + eps))

# SDRi Metric
class SDRiMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        classes = logits.argmax(dim=-1)
        return self.metric(classes, labels)


class SISNRi(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, logits, targets, mix, **kwargs):
        with torch.no_grad():
            sisnr = torch.maximum((si_snr(logits[:, 0], targets[:, 0]) + si_snr(logits[:, 1], targets[:, 1])) / 2, (si_snr(logits[:, 1], targets[:, 0]) + si_snr(logits[:, 0], targets[:, 1])) / 2)
            mixed = (si_snr(mix, targets[:, 0]) + si_snr(mix, targets[:, 1])) / 2
            improvement = sisnr - mixed
            return improvement.mean().item()


# PESQ Metric
class PESQ(BaseMetric):
    def __init__(self):
        pass

    def __call__(self, **batch):
        return super().__call__(**batch)

# STOI Metric
class STOI(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, **batch):
        return super().__call__(**batch)