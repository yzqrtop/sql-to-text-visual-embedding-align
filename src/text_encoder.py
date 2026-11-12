"""Text encoder branch of the semantic gap model (Module 2).

Wraps a causal language model (Qwen2.5-Coder-7B by default) to produce the
text hidden states H_q in R^{L_q x d} used as the Query in the cross-modal
attention step.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from . import config


class TextEncoder(nn.Module):
    """Encode a natural-language question into token-level hidden states.

    The paper uses Qwen2.5-Coder-7B; any HuggingFace causal LM can be
    substituted via ``config.TEXT_MODEL_NAME``.

    Outputs:
        H_q: tensor of shape (1, L_q, d) -- token hidden states.
    """

    def __init__(self, model_name: Optional[str] = None, freeze: bool = True):
        super().__init__()
        model_name = model_name or config.TEXT_MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.hidden_dim = self.model.config.hidden_size

    def forward(self, text: str) -> torch.Tensor:
        """Return token-level hidden states H_q for the given text."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.TEXT_MAX_LEN,
            padding=True,
        ).to(next(self.model.parameters()).device)
        with torch.set_grad_enabled(not self._frozen()):
            outputs = self.model(**enc, output_hidden_states=True)
        # Use the last hidden state; shape (batch, L_q, d)
        h_q = outputs.last_hidden_state
        return h_q

    def _frozen(self) -> bool:
        return not any(p.requires_grad for p in self.model.parameters())

    @property
    def output_dim(self) -> int:
        return self.hidden_dim
