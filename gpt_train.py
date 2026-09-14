import argparse
import torch
from gpt_model import GPTLanguageModel, device

parser = argparse.ArgumentParser()
parser.add_argument('--block_size', type=int, default=256, help='The maximum context length for the model.')
parser.add_argument('--n_embd', type=int, default=384, help='The dimensionality of the token embeddings.')
parser.add_argument('--n_head', type=int, default=6, help='The number of attention heads in each multi-head attention layer.')
parser.add_argument('--n_layer', type=int, default=6, help='The number of transformer blocks in the model.')
parser.add_argument('--dropout', type=float, default=0.2, help='The dropout rate for regularization.')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='The learning rate for the optimizer.')
parser.add_argument('--training_iterations', type=int, default=10000, help='The number of training iterations.')
parser.add_argument('--batch_size', type=int, default=256, help='The number of samples per batch during training.')
args = parser.parse_args()


learning_rate = args.learning_rate
training_iterations = args.training_iterations
batch_size = args.batch_size
block_size = args.block_size

def encode(string: str):
    result = []
    for ch in string:
        result.append(stoi[ch])
    return result

with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {}
itos = {}

for integ, ch in enumerate(chars):
    stoi[ch] = integ
    itos[integ] = ch

data = torch.tensor(encode(text), dtype=torch.long)     # Encode the whole text into a tensor of integers
train_size = int(0.9*len(data))
train_data = data[:train_size]
val_data = data[train_size:]

def get_batch(data_type: str):
    if data_type == 'training':
        data  = train_data
    else:
        data = val_data

    indices = torch.randint(len(data) - block_size, (batch_size,))
    inputs = []
    targets = []
    for i in indices:
        inputs.append(data[i:i+block_size])
        targets.append(data[i+1:i+1+block_size])
    stacked_inputs = torch.stack(inputs)
    stacked_targets = torch.stack(targets)

    stacked_inputs = stacked_inputs.to(device)
    stacked_targets = stacked_targets.to(device)

    return stacked_inputs, stacked_targets

gpt_model = GPTLanguageModel(
    vocab_size=len(chars),
    n_embd=args.n_embd,
    n_head=args.n_head,
    n_layer=args.n_layer,
    block_size=args.block_size,
    dropout=args.dropout,
)

model_device = gpt_model.to(device)
optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=learning_rate)

batches_to_avg = 50

def save_checkpoint(filename):
    checkpoint = {
        'model_state_dict': gpt_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stoi': stoi,
        'itos': itos,
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_embd': args.n_embd,
        'n_head': args.n_head,
        'n_layer': args.n_layer,
        'dropout': args.dropout
    }
    torch.save(checkpoint, filename)

@torch.no_grad()    # Disable gradient tracking for evaluation.
def estimate_loss():
    out = {}
    gpt_model.eval()    # Disable dropout, so the loss estimate is stable and reproducible.
    for split in ['training', 'validation']:
        losses = torch.zeros(batches_to_avg)
        for k in range(batches_to_avg):
            inputs, targets = get_batch(split)
            logits, loss = gpt_model(inputs, targets)    # Calls forward() internally.
            losses[k] = loss.item()
        out[split] = losses.mean()
    gpt_model.train()   # Re-enable dropout for the rest of the training.
    return out

best_validation_loss = float('inf')
for it in range(training_iterations):
    inputs, targets = get_batch('training')
    logits, loss = gpt_model(inputs, targets)    # Calls forward() internally.

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if it % 500 == 0:
        losses = estimate_loss()
        print(f"Step {it}: Training loss: {losses['training']:.4f}, Validation loss: {losses['validation']:.4f}", flush=True)
        save_checkpoint('gpt_latest_model.pt')
        if losses['validation'] < best_validation_loss:
            best_validation_loss = losses['validation']
            save_checkpoint('gpt_best_model.pt')   # Save the best model checkpoint based on validation loss.

save_checkpoint('gpt_latest_model.pt')
print("Latest model saved to gpt_latest_model.pt, best model saved to gpt_best_model.pt")