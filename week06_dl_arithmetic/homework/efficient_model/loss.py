"""
Cross Entropy Loss for Causal LM
"""

import torch
import torch.nn as nn

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


class CrossEntropyLoss(nn.Module):
    """Fused Linear Cross Entropy for causal LM."""
    # TODO: Replace with fused linear cross entropy (LigerFusedLinearCrossEntropyLoss)
    # The fused version takes hidden_states + lm_head.weight instead of logits

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.fused_liger_ce = LigerFusedLinearCrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, hidden_states, lm_head_weight, labels) -> torch.Tensor:
        # TODO: Implement forward pass
        shift_hidden = hidden_states[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        shift_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
        shift_labels = shift_labels.view(-1)

        return self.fused_liger_ce(lm_head_weight, shift_hidden, shift_labels)

