"""
Adapted self attention pooling
https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
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
        self.softmax = nn.functional.softmax
        
    def forward(self, batch_rep:torch.Tensor) -> torch.Tensor:
        """
        Run attention pooling

        :param batch_rep: torch.Tensor of batched output, size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        :return utter_rep:torch.Tensor, pooled output, size (N, H)
        """
        att_w = self.softmax(self.W(batch_rep).squeeze(-1), dim=1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep