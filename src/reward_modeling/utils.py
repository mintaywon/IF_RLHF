import random
import datasets
import numpy as np
import warnings
from typing import Dict
from dataclasses import dataclass
from typing import Optional
from trl import ModelConfig, RewardConfig, RewardTrainer
@dataclass
class MyRewardConfig(RewardConfig):
    """
    RewardConfig collects all training arguments related to the [`RewardTrainer`] class.
    MyRewardConfig is a subclass of RewardConfig that adds a length_bias_ratio argument.

    Parameters:
        length_bias_ratio (`float`, *optional*, defaults to `0.2`):
        data_dir (`str`, *optional*, defaults to None):
    """
    train_datapath: Optional[str] = None
    eval_datapath: Optional[str] = None

def compute_accuracy(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    # Check for ties in predictions
    ties = predictions[:, 0] == predictions[:, 1]
    tie_count = ties.sum()
    if tie_count > 0:
        warnings.warn(
            f"There are {tie_count} out of {len(predictions)} instances where the predictions for both options are equal. Adjusting accuracy accordingly."
        )
    else:
        print("No ties in predictions.")
    
    # Use argmax to select the higher reward prediction, considering ties result in selecting index 0
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Correct predictions are those where predicted labels match the actual labels
    correct_predictions = (predicted_labels == labels).astype(float)
    
    # Adjust for ties: For each tie, if the actual label is 0, subtract 0.5 from the correct predictions count,
    # since the tie was initially considered fully correct but should only be half correct.
    correct_predictions -= (ties & (labels == 0)).astype(float) * 0.5
    correct_predictions += (ties & (labels == 1)).astype(float) * 0.5
    
    # Compute the adjusted accuracy
    adjusted_accuracy = correct_predictions.mean().item()
    
    return {"accuracy": adjusted_accuracy}
