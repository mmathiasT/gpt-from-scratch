import torch
from gpt_model import GPTLanguageModel, block_size, device

learning_rate = 1e-3
training_iterations = 10000

batch_size = 256

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


gpt_model = GPTLanguageModel(vocab_size=len(chars))
model_device = gpt_model.to(device)
optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=learning_rate)

batches_to_avg = 50

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

for it in range(training_iterations):
    inputs, targets = get_batch('training')
    logits, loss = gpt_model(inputs, targets)    # Calls forward() internally.

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if it % 500 == 0:
        losses = estimate_loss()
        print(f"Step {it}: Training loss: {losses['training']:.4f}, Validation loss: {losses['validation']:.4f}")

torch.save({
    'model_state_dict': gpt_model.state_dict(),
    'stoi': stoi,
    'itos': itos,
    'vocab_size': vocab_size,
}, 'gpt_model.pt')
print("Model saved to gpt_model.pt")