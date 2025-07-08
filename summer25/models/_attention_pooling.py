"""
Adapted self attention pooling
https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362

Author(s): Hugo Botha, source, Daniela Wiepert
Last modified: 07/2025
"""
#IMPORTS
##third-party
import torch.nn as nn
import torch 

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf

    :param input_dim: int, hidden state size coming out of model (H)
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, x:torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Run attention pooling

        :param batch_rep: torch.Tensor of batched output, size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        :return utter_rep:torch.Tensor, pooled output, size (N, H)
        """
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        weights = self.W(x).squeeze(-1)  # [batch, time]
        weights[~mask.squeeze()] = float('-inf')  # mask padding
        attn_weights = torch.softmax(weights, dim=1)
        return (x * attn_weights.unsqueeze(-1)).sum(dim=1)