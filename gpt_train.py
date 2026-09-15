import argparse
import torch
import math
from gpt_model import GPTLanguageModel, device

parser = argparse.ArgumentParser()
parser.add_argument('--block_size', type=int, default=256, help='The maximum context length for the model.')
parser.add_argument('--n_embd', type=int, default=384, help='The dimensionality of the token embeddings.')
parser.add_argument('--n_head', type=int, default=6, help='The number of attention heads in each multi-head attention layer.')
parser.add_argument('--n_layer', type=int, default=6, help='The number of transformer blocks in the model.')
parser.add_argument('--dropout', type=float, default=0.2, help='The dropout rate for regularization.')
parser.add_argument('--max_learning_rate', type=float, default=1e-3, help='The learning rate for the optimizer.')
parser.add_argument('--training_iterations', type=int, default=10000, help='The number of training iterations.')
parser.add_argument('--batch_size', type=int, default=256, help='The number of samples per batch during training.')
parser.add_argument('--warmup_iterations', type=int, default=1000, help='The number of iterations to linearly increase the learning rate before decaying.')
parser.add_argument('--min_learning_rate', type=float, default=None, help="Minimum learning rate at the end of cosine decay. Defaults to learning_rate / 10.")
args = parser.parse_args()


max_learning_rate = args.max_learning_rate
training_iterations = args.training_iterations
warmup_iterations = args.warmup_iterations
batch_size = args.batch_size
block_size = args.block_size

if args.min_learning_rate is None:
    min_learning_rate = max_learning_rate / 10
else:
    min_learning_rate = args.min_learning_rate


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
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]


def get_batch(data_type: str):
    if data_type == 'training':
        data = train_data
    else:
        data = val_data

    indices = torch.randint(len(data) - block_size, (batch_size,))
    inputs = []
    targets = []
    for i in indices:
        inputs.append(data[i:i + block_size])
        targets.append(data[i + 1:i + 1 + block_size])
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
optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=max_learning_rate)

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


def get_learning_rate(iteration: int, max_lr: float, min_lr: float, warmup_iterations: int, training_iterations: int):
    if iteration < warmup_iterations:
        return max_lr * (iteration / warmup_iterations)
    else:
        decay_ratio = (iteration - warmup_iterations) / (training_iterations - warmup_iterations)
        # Cosine of pi*decay_ratio is going from 1 to -1 as decay_ratio goes from 0 to 1. We add 1 and divide by 2 so it goes from 1 to 0.
        coeff = (1 + math.cos(math.pi * decay_ratio)) / 2
        return min_lr + coeff * (max_lr - min_lr)


best_validation_loss = float('inf')
for it in range(training_iterations):
    inputs, targets = get_batch('training')
    logits, loss = gpt_model(inputs, targets)    # Calls forward() internally.

    learning_rate = get_learning_rate(it, max_learning_rate, min_learning_rate, warmup_iterations, training_iterations)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if it % 500 == 0:
        losses = estimate_loss()
        print(f"Step {it}: Training loss: {losses['training']:.4f}, Validation loss: {losses['validation']:.4f}, LR: {learning_rate:.6f}", flush=True)
        save_checkpoint('gpt_latest_model.pt')
        if losses['validation'] < best_validation_loss:
            best_validation_loss = losses['validation']
            save_checkpoint('gpt_best_model.pt')   # Save the best model checkpoint based on validation loss.

final_losses = estimate_loss()
print(f"Final (step {training_iterations}): Training loss: {final_losses['training']:.4f}, Validation loss: {final_losses['validation']:.4f}", flush=True)
save_checkpoint('gpt_latest_model.pt')
if final_losses['validation'] < best_validation_loss:
    save_checkpoint('gpt_best_model.pt')
print("Latest model saved to gpt_latest_model.pt, best model saved to gpt_best_model.pt")
