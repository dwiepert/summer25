"""
Custom ranked classification loss

Author(s): Leland Barnard
Last modified: 06/2025
"""
#IMPORTS
##third-party
import torch
import torch.nn as nn
 
class RankedClassificationLoss(nn.Module):
    """
    Ranked classification loss class

    :param rating_threshold: float, value for determining whether a rating (rank) is 1 or 0
    :param margin: float, margin value for nn.MarginRankingLoss (default = 1.0)
    :param bce_weight: float, weight to apply to BCE loss (default = 0.5)
    """
    def __init__(self, rating_threshold:float, margin:float=1.0, bce_weight:float=0.5):
        super().__init__()
        self.rating_threshold = rating_threshold
        self.margin = margin
        self.bce_weight = bce_weight
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.mr_criterion = nn.MarginRankingLoss(margin=self.margin)
       
    def forward(self, logits: torch.Tensor, ratings:torch.Tensor) -> torch.Tensor:
        """
        Calculate loss 

        :param logits: torch.Tensor, model prediction (N, # classes)
        :param ratings: torch.Tensor, target ratings (N, # classes)
        :return: torch.Tensor, calculated loss
        """
        # Convert ratings to binary labels for BCE loss
        binary_labels = (ratings >= self.rating_threshold).float()
        mr_loss_entries = []
        # Iterate over each target to calculate pairwise margin ranking loss
        for c in range(ratings.shape[1]):
            # Get all unique pairs of ratings and logits for the current target
            rating_pairs = torch.combinations(ratings[:, c])
            logit_pairs = torch.combinations(logits[:, c])
 
            # Determine the target ranking (-1, 0, or 1)
            # y = 1 means x1 should be > x2
            # y = -1 means x2 should be > x1
            y = torch.sign(rating_pairs[:, 0] - rating_pairs[:, 1])
 
            x1 = logit_pairs[:, 0]
            x2 = logit_pairs[:, 1]
 
            # Calculate and store the margin ranking loss for the current target
            l = self.mr_criterion(x1, x2, y)
            mr_loss_entries.append(l)
 
        # Stack the losses to keep them in the computation graph, then take the mean
        if mr_loss_entries:
            mr_loss = torch.stack(mr_loss_entries).mean()
        else:
            mr_loss = torch.tensor(0.0, device=logits.device)
 
        # Calculate standard BCE loss
        bce_loss = self.bce_criterion(logits, binary_labels)
       
        # Return the weighted average of the two losses
        return (1 - self.bce_weight) * mr_loss + self.bce_weight * bce_loss