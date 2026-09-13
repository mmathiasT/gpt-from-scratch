import torch
from gpt_model import GPTLanguageModel, device

def decode(indices: list):
    result = []
    for i in indices:
        result.append(gpt_checkpoint['itos'][i])
    return ''.join(result)

gpt_checkpoint = torch.load('gpt_model.pt', map_location=device)

gpt_model = GPTLanguageModel(gpt_checkpoint['vocab_size'])
gpt_model.load_state_dict(gpt_checkpoint['model_state_dict'])       # Overwrite the randomly initialized weights with the trained weights from the checkpoint.
gpt_model.to(device)
gpt_model.eval()

start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = gpt_model.generate(start_context, max_new_tokens=100)
print(decode(generated_text[0].tolist()))