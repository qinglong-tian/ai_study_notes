# model_module.py (workshop version: simple, readable, PyTorch built-ins)
import torch
import torch.nn as nn


class VectorICLTransformer(nn.Module):
    """
    A small GPT-style (decoder-only) Transformer for in-context learning on vector tokens.

    Input:
        tokens: (B, T, d_in) where T = 2k+1 and tokens are
                [x1, y1, x2, y2, ..., xk, yk, x_query]
        (y tokens are vectors with y in coordinate 0 and zeros elsewhere)

    Output:
        preds:  (B, k+1) predictions at x-positions: indices 0,2,4,...,2k
               so it lines up with targets = [y1, ..., yk, y_query]
    """

    def __init__(
        self,
        d_in=20,
        d_model=256,
        n_layers=12,
        n_heads=8,
        dropout=0.0,
        max_seq_len=512,
    ):
        super().__init__()

        self.d_in = d_in
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # turn each d_in vector token into a d_model embedding
        self.in_proj = nn.Linear(d_in, d_model)

        # learned positional embeddings
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks (PyTorch built-in)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm, stable for training
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

        # scalar prediction head
        self.out = nn.Linear(d_model, 1)

    def forward(self, tokens, k_pairs=None):
        """
        tokens: (B, T, d_in)
        returns: (B, k+1)
        """
        B, T, d_in = tokens.shape

        # (optional) minimal sanity check to catch obvious mistakes
        if d_in != self.d_in:
            raise ValueError("tokens last dim must equal d_in")

        if T > self.max_seq_len:
            raise ValueError("sequence too long; increase max_seq_len")

        # add positional embeddings
        h = self.in_proj(tokens)                            # (B,T,d_model)
        pos = torch.arange(T, device=tokens.device)         # (T,)
        h = h + self.pos_emb(pos)[None, :, :]               # (B,T,d_model)

        # causal mask: each position can only attend to earlier positions
        causal_mask = torch.triu(
            torch.ones((T, T), device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )

        h = self.tr(h, mask=causal_mask)                    # (B,T,d_model)
        h = self.ln(h)

        # take outputs only at x-positions: 0,2,4,...,2k
        x_pos = torch.arange(0, T, 2, device=tokens.device)  # length = k+1
        h_x = h[:, x_pos, :]                                 # (B,k+1,d_model)

        preds = self.out(h_x).squeeze(-1)                    # (B,k+1)
        return preds


if __name__ == "__main__":
    # quick sanity check with the workshop data_module
    from data_module import InContextDataGenerator

    device = "cpu"
    gen = InContextDataGenerator(d=20, device=device, seed=0)
    batch = gen.sample_batch("linear", batch_size=2, k=11, effective_dim=5)

    model = VectorICLTransformer(d_in=20, d_model=128, n_layers=2, n_heads=4, max_seq_len=256).to(device)
    preds = model(batch.tokens, k_pairs=11)

    print("tokens:", batch.tokens.shape)
    print("preds: ", preds.shape)
    print("tgt:   ", batch.targets.shape)
