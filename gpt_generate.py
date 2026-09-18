import argparse

import torch
from gpt_model import GPTLanguageModel, device


def decode(indices: list):
    result = []
    for i in indices:
        result.append(gpt_checkpoint['itos'][i])
    return ''.join(result)


parser = argparse.ArgumentParser()
parser.add_argument('--version', choices=['latest', 'best'], default='best', help='Choose which model checkpoint to load: latest or best.')
parser.add_argument('--max_new_tokens', type=int, default=100, help='The maximum number of new tokens to generate.')
parser.add_argument('--temperature', type=float, default=1.0, help='The temperature for sampling. Higher values lead to more random outputs.')
parser.add_argument('--top_p', type=float, default=1.0, help='The top-p filtering parameter for nucleus sampling.')
args = parser.parse_args()

if args.version == 'latest':
    file_name = 'gpt_latest_model.pt'
else:
    file_name = 'gpt_best_model.pt'

gpt_checkpoint = torch.load(file_name, map_location=device)

gpt_model = GPTLanguageModel(gpt_checkpoint['vocab_size'], gpt_checkpoint['n_embd'], gpt_checkpoint['n_head'], gpt_checkpoint['n_layer'], gpt_checkpoint['block_size'], gpt_checkpoint['dropout'])
gpt_model.load_state_dict(gpt_checkpoint['model_state_dict'])       # Overwrite the randomly initialized weights with the trained weights from the checkpoint.
gpt_model.to(device)
gpt_model.eval()

start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    generated_text = gpt_model.generate(start_context, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
print(decode(generated_text[0].tolist()))
