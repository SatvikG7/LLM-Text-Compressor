import torch

def decompress(ranks, input_ids, tokenizer, model, M = 4) -> str:
    decompressed_ids = []
    window = input_ids.tolist()  # Start with the initial M tokens
    decompressed_ids.extend(window)
    with torch.no_grad():
      for i, rank in enumerate(ranks):
          print(f"Decompressing: {i}/{len(ranks)}", end="\r")
          input_window = torch.tensor([window]).to("cuda")
          outputs = model(input_window)
          predictions = outputs.logits
          next_token_logits = predictions[0, -1, :]
          sorted_logits = torch.argsort(next_token_logits, descending=True)
          next_token = sorted_logits[rank].item()
          decompressed_ids.append(next_token)
          window = window[1:] + [next_token]  # Update the sliding window

    return tokenizer.decode(decompressed_ids)
