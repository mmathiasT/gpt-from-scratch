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

A trained model (`gpt_model.pt`) is already included in this repo, so you can generate text right away, with no training needed:

```bash
python gpt_generate.py
```

This loads `gpt_model.pt` and prints 100 characters of generated text.

If you want to retrain the model yourself (e.g. after changing a setting), run:

```bash
python gpt_train.py
```

This trains the model from scratch, printing the training/validation loss every 500 steps, then overwrites `gpt_model.pt` with the newly trained weights (and the character vocabulary).

## Example output

```
It thou perdon strengtter, ever to she is and love,
You have see watch me.

DUKE OF YORK:
Welcome.
```

(Trained with a small setup — `n_embd=64`, `n_layer=4`, `n_head=4`, `block_size=64` — to keep training time reasonable on a CPU. Bigger settings and more training steps give more coherent, more word-like text.)

## Settings

The model's architecture settings live at the top of `gpt_model.py`. Training settings (learning rate, number of training steps, batch size) live at the top of `gpt_train.py`.

| Setting | What it controls |
|---|---|
| `block_size` | how many previous characters the model can look at |
| `n_embd` | how many numbers describe each character |
| `n_head` | how many attention "heads" work in parallel in each block |
| `n_layer` | how many Transformer blocks are stacked |
| `dropout` | how much randomness is used during training to avoid overfitting |

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
