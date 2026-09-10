import torch

learning_rate = 1e-3
training_iterations = 1000

block_size = 8
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'     # Use GPU if available, otherwise use CPU.

def encode(string: str):
    result = []
    for ch in string:
        result.append(stoi[ch])
    return result

def decode(indices: list):
    result = []
    for i in indices:
        result.append(itos[i])
    return ''.join(result)


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

class BigramLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)

    # Predicts logits for the next character based only on the current character.
    def forward(self, context, targets=None):  
        logits = self.token_embedding_table(context)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            B, T = targets.shape
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, context, max_new_tokens):
        for i in range(0,max_new_tokens):
            logits, loss = self(context)    # Calls forward() internally.
            # Copy predictions only for the last position.
            last_logits = logits[:, -1, :]  
            # Convert logits to probabilities, set dim to -1 to normalize across the vocab so each row sums to 1.
            probabilities = torch.nn.functional.softmax(last_logits, dim=-1) 
            next_chars = torch.multinomial(probabilities, num_samples=1) # Draw 1 character.
            context = torch.cat((context, next_chars), dim=1) # Append the newly sampled character to the end of each sequence.
        return context

bigram_model = BigramLanguageModel()
model_device = bigram_model.to(device)
optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=learning_rate)

for it in range(training_iterations):
    inputs, targets = get_batch('training')
    logits, loss = bigram_model(inputs, targets)    # Calls forward() internally.

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if it % 1000 == 0:
        print(loss.item())

bigram_model.eval()  # Disable dropout (none here, but keeps behavior consistent with gpt.py).
start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = bigram_model.generate(start_context, max_new_tokens=300)
print(decode(generated[0].tolist()))