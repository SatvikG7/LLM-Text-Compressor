from typing import List
import torch

def compress(input_ids, model, M = 4) -> List[int]:
    # Tokenize the input text
    ranks = []
    window_size = M  # M value in the paper (memory window size)
    with torch.no_grad():
        # Iterate over the text with the sliding window
        for i in range(len(input_ids[0]) - window_size):
            print(f"Compressing: {i}/{len(input_ids[0]) - window_size}", end="\r")
            # Get the tokens within the window
            input_window = input_ids[:, i:i + window_size]
            # Actual next word token (we want to predict this)
            true_next_token = input_ids[0, i + window_size]
            # Get model outputs for the current window
            outputs = model(input_window)
            predictions = outputs.logits
            # Get the predicted logits for the next token in the sequence
            next_token_logits = predictions[0, -1, :]
            # Compute the rank of the true next token among the predictions
            sorted_logits = torch.argsort(next_token_logits, descending=True)
            rank = (sorted_logits == true_next_token).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)
            # Print the predicted token (optional for debugging)
            # predicted_token = sorted_logits[0].item()
            # predicted_word = tokenizer.decode([predicted_token])
            # actual_word = tokenizer.decode([true_next_token])
            # print(f"Predicted: {predicted_word}, Actual: {actual_word}, Rank: {rank}")
    return ranks