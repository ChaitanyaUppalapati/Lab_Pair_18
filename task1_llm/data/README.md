# TinyStories (shared raw data)

Source: https://huggingface.co/datasets/roneneldan/TinyStories

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")
```

Raw files are gitignored. Each member creates their **own** 100K train / 10K validation split and records the seed in their config.
