import torch

learning_rate = 1e-3
training_iterations = 10000

block_size = 64
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'     # Use GPU if available, otherwise use CPU.

n_head = 4  # Number of attention heads in the multi-head attention mechanism.
n_layer = 4  # Number of transformer blocks in the model.
n_embd = 64  # Size of the embedding vector for each token.
dropout = 0.2  # Dropout rate for regularization, so the model doesn't overfit to the specific pieces of training data .

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

class Head(torch.nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # bias=False on all three, so they don't have constant offsets.
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)

        # Lower triangular mask: position i can only attend to positions with indices less than or equal to i.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Dropout layer to prevent overlearning.
        self.dropout = torch.nn.Dropout(dropout)

    # For each token, gather a weighted mix of value vectors from itself and earlier tokens.
    def forward(self, token_repr):
        batch_size, num_tokens, embedding_dim = token_repr.shape
        # Compute queries, keys for all tokens in the batch. The shape of the output is (batch_size, num_tokens, head_size).
        k = self.key(token_repr)
        q = self.query(token_repr)

        k_tr = k.transpose(-2, -1)
        attention_weights = (q @ k_tr) / k.size(-1)**0.5
        masking_matrix = torch.zeros(num_tokens, num_tokens).masked_fill(self.tril[:num_tokens, :num_tokens] == 0, float('-inf'))

        masked = attention_weights.masked_fill(masking_matrix == float('-inf'), float('-inf'))
        softmaxed = torch.nn.functional.softmax(masked, dim=-1)

        dropouted = self.dropout(softmaxed)
        
        return dropouted @ self.value(token_repr)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_head: int, head_size: int):
        super().__init__()
        self.heads = torch.nn.ModuleList()
        for _ in range(num_head):
            self.heads.append(Head(head_size))
        self.proj = torch.nn.Linear(head_size * num_head, n_embd)
        self.dropout = torch.nn.Dropout(dropout)

    # Run all attention heads in parallel on the same input, then merge their outputs.
    def forward(self, token_repr):
        out = []
        for h in self.heads:
            out.append(h(token_repr)) # Run each head (call + forward) on the same input and collect their outputs.

        concatenated = torch.cat(out, dim=-1)
        projected = self.proj(concatenated)     # Project the concatenated outputs back to n_embd dimensions.
        return self.dropout(projected)

class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = torch.nn.Linear(n_embd, 4 * n_embd)     # Expand the representation, so the non-linear function has more space to work with.
        self.activation = torch.nn.ReLU()                     # Not linear function for universal approximation theorem.
        self.shrink = torch.nn.Linear(4 * n_embd, n_embd)     # Weighted sum of the 4*n_embd ReLU "pieces" back down to n_embd. Approximation of some function of the token repr.
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, token_repr):
        return self.dropout(self.shrink(self.activation(self.expand(token_repr))))

class Block(torch.nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attention = MultiHeadAttention(num_head=n_head, head_size=n_embd//n_head)
        self.feed_forward = FeedForward()
        self.layer_norm1 = torch.nn.LayerNorm(n_embd)
        self.layer_norm2 = torch.nn.LayerNorm(n_embd)

    def forward(self, token_repr):  
        token_repr = token_repr + self.attention(self.layer_norm1(token_repr))  # Residual connection.
        token_repr = token_repr + self.feed_forward(self.layer_norm2(token_repr))  # Residual connection.
        return token_repr

class GPTLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embd)  # Similar to token embeddings, but for positions.
        self.final_linear_layer = torch.nn.Linear(n_embd, vocab_size)
        self.blocks = torch.nn.ModuleList()
        for _ in range(n_layer):
            self.blocks.append(Block(n_embd, n_head))
        self.final_normalization = torch.nn.LayerNorm(n_embd)
        
    # Predicts logits for the next character, using the full preceding context (up to block_size tokens).
    def forward(self, context, targets=None):
        batch_size, num_tokens = context.shape
        tok_emb = self.token_embedding_table(context)
        position_emb = self.position_embedding_table(torch.arange(num_tokens, device=device))
        token_repr = tok_emb + position_emb
        for block in self.blocks:
            token_repr = block(token_repr)
        token_repr = self.final_normalization(token_repr)
        logits = self.final_linear_layer(token_repr)


        loss = None
        if targets is not None:
            batch_size, num_tokens, vocab_size = logits.shape
            logits = logits.view(batch_size * num_tokens, vocab_size)
            batch_size, num_tokens = targets.shape
            targets = targets.view(batch_size * num_tokens)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, context, max_new_tokens):
        for i in range(0,max_new_tokens):
            logits, loss = self(context[:, -block_size:])    # Calls forward() internally.
            # Copy predictions only for the last position.
            last_logits = logits[:, -1, :]  
            # Convert logits to probabilities, set dim to -1 to normalize across the vocab so each row sums to 1.
            probabilities = torch.nn.functional.softmax(last_logits, dim=-1) 
            next_chars = torch.multinomial(probabilities, num_samples=1) # Draw 1 character.
            context = torch.cat((context, next_chars), dim=1) # Append the newly sampled character to the end of each sequence.
        return context

gpt_model = GPTLanguageModel()
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

gpt_model.eval()
start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = gpt_model.generate(start_context, max_new_tokens=100)
print(decode(generated_text[0].tolist()))