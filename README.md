# Character-level GPT

A small GPT-style language model built from scratch in PyTorch. It learns to write text one character at a time, trained on Shakespeare's plays.

## What it does

The model reads text character by character and learns to guess the next character. After training, it can write new text on its own, one character at a time, in a style close to the training text.

## How it works

- Each character and its position get turned into a list of numbers (embeddings).
- The numbers pass through several Transformer blocks. Each block has two parts:
  - Self-attention: lets each character look back at earlier characters and pick up context.
  - A small feedforward network: processes each character's numbers a bit further.
- A final layer turns the numbers into a guess for the next character.

## Setup

```bash
pip install -r requirements.txt
```

The training text (`input.txt`, Shakespeare's plays) is already included in this repo.

## Usage

Trained model checkpoints (`gpt_best_model.pt`, `gpt_latest_model.pt`) are already included in this repo, so you can generate text right away, with no training needed:

```bash
python gpt_generate.py
```

This loads `gpt_best_model.pt` by default and prints 100 characters of generated text.

Optional flags:

```bash
python gpt_generate.py --version latest --max_new_tokens 300 --temperature 0.8 --top_p 0.9
```

| Flag | What it controls |
|---|---|
| `--version` | which checkpoint to load: `best` (lowest validation loss seen) or `latest` (most recent step) |
| `--max_new_tokens` | how many characters to generate |
| `--temperature` | randomness of sampling — lower is more predictable, higher is more random |
| `--top_p` | nucleus sampling threshold — only sample from the smallest set of characters whose combined probability exceeds this value |

If you want to retrain the model yourself, run:

```bash
python gpt_train.py
```

This trains the model from scratch, printing the training/validation loss every 500 steps. It saves two checkpoints as it goes: `gpt_latest_model.pt` (updated every 500 steps) and `gpt_best_model.pt` (updated only when validation loss improves), so training can be interrupted at any time without losing progress. Architecture and training settings can be overridden with flags, e.g.:

```bash
python gpt_train.py --n_embd 256 --n_head 4 --n_layer 4 --block_size 128 --batch_size 64 --training_iterations 5000
```

## Example output

```
KING RICLARD II:
Our sin;' then, that fly harm
the noble receive hed.

AUTOLYCUS:
O, thou used upon these throne!

COMINIUS:
Fear prince, sir. Poor to hearly it.
```

(Generated with `--temperature 0.8 --top_p 0.9`. Model trained with `n_embd=256`, `n_head=4`, `n_layer=4`, `block_size=128`, `batch_size=64`, `dropout=0.1`, for up to 5000 training steps — several hours on a CPU.)

## Settings

Architecture and training settings are passed as command-line flags to `gpt_train.py` (see `python gpt_train.py --help` for defaults).

| Setting | What it controls |
|---|---|
| `--block_size` | how many previous characters the model can look at |
| `--n_embd` | how many numbers describe each character |
| `--n_head` | how many attention "heads" work in parallel in each block |
| `--n_layer` | how many Transformer blocks are stacked |
| `--dropout` | how much randomness is used during training to avoid overfitting |
| `--learning_rate` | how big each optimizer update step is |
| `--batch_size` | how many sequences are processed per training step |
| `--training_iterations` | how many training steps to run |

## Simpler baseline: `bigram.py`

`bigram.py` is a much simpler model — a bigram model that only looks at the single previous character to guess the next one, with no attention and no Transformer blocks. It's included to show the starting point before adding attention, and to make the jump in quality from `gpt_model.py`/`gpt_train.py` easier to see.

## Sources

Built while learning about Transformers and attention mechanisms from:

- [Transformers in Deep Learning Course](https://www.youtube.com/watch?v=lRylkiFdUdk&list=PLuhqtP7jdD8CQTxwVsuiFYGvHtFpNhlR3)
- [Andrej Karpathy - Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [GPT Architecture | How to create ChatGPT from Scratch?](https://www.youtube.com/watch?v=kNxGURHpYqM)
- [Multi-Head Attention Explained Visually | Simple Transformer Guide](https://www.youtube.com/watch?v=42L1q1Z4Ojc)
- [How GPT Actually Works: Transformer Decoder Explained Visually](https://www.youtube.com/watch?v=KE9fqU4EG4o)
- [Understanding GPT: A Simple Explanation of Its Architecture and Applications](https://ravjot03.medium.com/understanding-gpt-a-simple-explanation-of-its-architecture-and-applications-94ef2b92b172)
- [Generative pre-trained transformer - Wikipedia](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer)
