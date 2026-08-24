"""LOUPE-style learnable, differentiable k-space sampling mask (B1, the MICCAI method).

A learnable 1D phase-encode mask whose probabilities are trained end-to-end for the downstream task (segmentation),
not for image fidelity. Budget-constrained to acceleration R (expected sampled lines = N/R), ACS center forced on.
Gradients flow from the task loss -> the mask logits, so the acquisition adapts to what segmentation needs.

Ref: Bahadir, Dalca, Sabuncu, "Learning-based Optimization of the Under-sampling Pattern in MRI" (IPMI'19/TCI'20).
"""
import torch
import torch.nn as nn


def rescale_probs(p, target, iters=8):
    """Renormalize p in [0,1] so mean(p) ~= target (LOUPE renorm: scale down, or scale up the complement)."""
    for _ in range(iters):
        m = p.mean()
        if m > target:
            p = p * (target / (m + 1e-8))
        else:
            p = 1.0 - (1.0 - p) * ((1.0 - target) / (1.0 - m + 1e-8))
        p = p.clamp(1e-4, 1 - 1e-4)
    return p


class LOUPEMask(nn.Module):
    def __init__(self, n_lines, R, slope=5.0, tau=0.5, acs_frac=0.08):
        super().__init__()
        self.n = n_lines
        self.R = R
        self.slope = slope
        self.tau = tau
        self.logits = nn.Parameter(torch.zeros(n_lines))     # learnable, init -> uniform p=0.5
        c = n_lines // 2
        n_acs = max(int(round(n_lines * acs_frac)), 4)
        acs = torch.zeros(n_lines)
        acs[c - n_acs // 2: c + (n_acs - n_acs // 2)] = 1.0
        self.register_buffer("acs", acs)

    def probs(self):
        p = torch.sigmoid(self.slope * self.logits)
        # budget excludes the always-on ACS lines
        target = max((self.n / self.R - self.acs.sum().item()), 1.0) / (self.n - self.acs.sum().item())
        p = rescale_probs(p, target)
        return torch.maximum(p, self.acs)                    # ACS forced to 1

    def forward(self, training=True):
        """Return a (relaxed, if training) binary mask of length n."""
        p = self.probs()
        if training:
            u = torch.rand_like(p).clamp(1e-6, 1 - 1e-6)
            # binary-concrete / relaxed Bernoulli with logits = logit(p)
            logit_p = torch.log(p) - torch.log(1 - p)
            m = torch.sigmoid((logit_p + torch.log(u) - torch.log(1 - u)) / self.tau)
        else:
            k = int(round(self.n / self.R))                  # deploy: top-(N/R) lines (ACS have p=1 -> included)
            idx = torch.topk(p, k).indices
            m = torch.zeros_like(p); m[idx] = 1.0
        return torch.maximum(m, self.acs)                    # ACS always sampled


def undersample(img, mask1d, pe_axis=-2):
    """Differentiable retrospective undersampling of a magnitude image (B,1,H,W) along pe_axis.
    mask1d: (n,) over the phase-encode axis. Returns |IFFT(mask * FFT(img))|."""
    k = torch.fft.fftshift(torch.fft.fft2(img.to(torch.complex64)), dim=(-2, -1))
    shape = [1] * img.dim(); shape[pe_axis] = mask1d.numel()
    k = k * mask1d.view(shape)
    return torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1))).abs()


def self_test():
    print("=== LOUPE self-test ===")
    N, R = 256, 8
    mask = LOUPEMask(N, R)
    p = mask.probs()
    print(f"expected sampled fraction = {p.mean().item():.3f}  (target 1/R = {1/R:.3f})  "
          f"eff_R ~= {1/p.mean().item():.1f}")
    print(f"ACS lines forced on: {(p[mask.acs.bool()] > 0.99).all().item()}")
    # gradient flow: undersample a dummy image, fake seg loss, backprop to logits
    img = torch.rand(2, 1, N, N)
    m = mask(training=True)
    rec = undersample(img, m)
    loss = ((rec - img) ** 2).mean()      # placeholder; real B1 uses seg loss
    loss.backward()
    g = mask.logits.grad
    print(f"gradient flows to mask logits: {g is not None and g.abs().sum().item() > 0}  "
          f"(|grad| sum = {g.abs().sum().item():.3e})")
    hard = mask(training=False)
    print(f"hard mask sampled lines = {int(hard.sum().item())} / {N}  (budget N/R = {N//R})")
    ok = (abs(p.mean().item() - 1/R) < 0.03) and (g.abs().sum().item() > 0)
    print(f"SELF TEST {'PASS' if ok else 'CHECK'}: budget met + gradients flow")
    return ok


if __name__ == "__main__":
    self_test()
