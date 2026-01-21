# data_module.py (workshop version: linear + sparse linear only, no meta)
from dataclasses import dataclass
import torch


@dataclass
class PromptBatch:
    tokens: torch.Tensor   # (B, 2k+1, d)
    targets: torch.Tensor  # (B, k+1)


class InContextDataGenerator:
    """
    Generate synthetic in-context learning prompts on the fly (linear + sparse linear).

    Prompt tokens:
        [x1, y1, x2, y2, ..., xk, yk, x_query]   length = 2k+1
    where each y-token is a d-dim vector with y stored in coordinate 0.

    Targets:
        [y1, y2, ..., yk, y_query]              length = k+1

    Curriculum learning via effective_dim:
        If effective_dim=m, then coordinates m..d-1 of all x are set to 0.
        For sparse_linear, the nonzero support is chosen among the first m coordinates.
    """

    def __init__(self, d=20, device="cpu", seed=0):
        self.d = d
        self.device = torch.device(device)
        self.g = torch.Generator(device=self.device)
        self.g.manual_seed(seed)

    def sample_batch(self, task: str, batch_size: int, k: int,
                     effective_dim=None, sparse_s: int = 3) -> PromptBatch:
        """
        task: "linear" or "sparse_linear"
        """
        if task == "linear":
            return self._linear(batch_size, k, effective_dim)

        if task == "sparse_linear":
            return self._sparse_linear(batch_size, k, effective_dim, sparse_s)

        raise ValueError(f"Unknown task: {task}")

    # ---------------- helpers ----------------
    def randn(self, *shape):
        return torch.randn(shape, generator=self.g, device=self.device)

    def mask_dim(self, x, effective_dim):
        """Zero out coordinates >= effective_dim."""
        if effective_dim is None:
            return x
        x = x.clone()
        x[..., effective_dim:] = 0.0
        return x

    def pack_prompt(self, x, y, x_query):
        """
        x: (B,k,d), y: (B,k), x_query: (B,d)
        returns tokens: (B,2k+1,d)
        """
        B, k, d = x.shape

        y_tok = torch.zeros((B, k, d), device=self.device)
        y_tok[:, :, 0] = y

        tokens = torch.empty((B, 2 * k + 1, d), device=self.device)
        tokens[:, 0:2 * k:2, :] = x
        tokens[:, 1:2 * k:2, :] = y_tok
        tokens[:, 2 * k, :] = x_query
        return tokens

    # ---------------- tasks ----------------
    def _linear(self, B, k, effective_dim):
        d = self.d

        # sample a random linear function: f(x) = w^T x
        w = self.randn(B, d)

        # sample in-context inputs and a query
        x = self.randn(B, k, d)
        xq = self.randn(B, d)

        # curriculum: keep only first effective_dim coordinates
        x = self.mask_dim(x, effective_dim)
        xq = self.mask_dim(xq, effective_dim)

        # outputs
        y = torch.einsum("bkd,bd->bk", x, w)
        yq = torch.einsum("bd,bd->b", xq, w)

        tokens = self.pack_prompt(x, y, xq)
        targets = torch.cat([y, yq[:, None]], dim=1)
        return PromptBatch(tokens=tokens, targets=targets)

    def _sparse_linear(self, B, k, effective_dim, s):
        d = self.d
        ed = effective_dim if effective_dim is not None else d

        # sample w then keep only s random coordinates among first ed dims
        w = self.randn(B, d)
        w[:, ed:] = 0.0  # keep weights inside active subspace

        mask = torch.zeros((B, d), device=self.device)
        for b in range(B):
            idx = torch.randperm(ed, generator=self.g, device=self.device)[:s]
            mask[b, idx] = 1.0
        w = w * mask

        x = self.randn(B, k, d)
        xq = self.randn(B, d)

        x = self.mask_dim(x, effective_dim)
        xq = self.mask_dim(xq, effective_dim)

        y = torch.einsum("bkd,bd->bk", x, w)
        yq = torch.einsum("bd,bd->b", xq, w)

        tokens = self.pack_prompt(x, y, xq)
        targets = torch.cat([y, yq[:, None]], dim=1)
        return PromptBatch(tokens=tokens, targets=targets)


if __name__ == "__main__":
    gen = InContextDataGenerator(d=20, device="cpu", seed=0)

    batch = gen.sample_batch("linear", batch_size=4, k=10, effective_dim=5)
    print("linear:", batch.tokens.shape, batch.targets.shape)

    batch = gen.sample_batch("sparse_linear", batch_size=4, k=10, effective_dim=7, sparse_s=3)
    print("sparse_linear:", batch.tokens.shape, batch.targets.shape)
