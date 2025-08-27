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

        # Initialize a tensor to accumulate the margin ranking loss
        total_mr_loss = torch.tensor(0.0, device=logits.device)
        num_targets = ratings.shape[1]

        # Iterate over each target to calculate pairwise margin ranking loss
        for c in range(num_targets):
            # If batch size is less than 2, we can't create pairs.
            if ratings.shape[0] < 2:
                continue

            # Get all unique pairs of ratings and logits for the current target
            rating_pairs = torch.combinations(ratings[:, c])
            logit_pairs = torch.combinations(logits[:, c])
 
            # Determine the target ranking (1 or -1)
            y = torch.sign(rating_pairs[:, 0] - rating_pairs[:, 1])

            # Filter out pairs with the same rating (y=0)
            valid_pairs_mask = y != 0
            if not valid_pairs_mask.any():
                continue # No valid pairs in this batch for this target

            x1 = logit_pairs[valid_pairs_mask, 0]
            x2 = logit_pairs[valid_pairs_mask, 1]
            y_filtered = y[valid_pairs_mask]

            # Calculate and accumulate the margin ranking loss for the current target
            total_mr_loss += self.mr_criterion(x1, x2, y_filtered)
            # Average the loss over the number of targets
        mr_loss = total_mr_loss / num_targets if num_targets > 0 else total_mr_loss

        # Calculate standard BCE loss
        bce_loss = self.bce_criterion(logits, binary_labels)

       
        # Return the weighted average of the two losses
        return (1 - self.bce_weight) * mr_loss + self.bce_weight * bce_loss