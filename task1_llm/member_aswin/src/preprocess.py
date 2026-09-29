"""Create an independent character-level TinyStories split and fixed-length pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from common import REPO_ROOT, load_config, output_paths, set_seed, write_json

SMOKE_STORIES = [
    "Once upon a time, a small fox found a red ball. The fox shared it with a kind bird.",
    "Lily found a little seed. She planted it, gave it water, and a yellow flower grew.",
    "The dragon was afraid of the dark, so his friend Mia brought a tiny lantern.",
    "Ben made a paper boat. It sailed across the pond and stopped beside a sleepy frog.",
    "A blue kite flew over the hill. Sam held the string and laughed with his sister.",
    "The old cat heard a sound in the garden. It was only a mouse eating a berry.",
    "Nora baked three cakes for the party. Everyone sang, danced, and helped clean up.",
    "A brave rabbit crossed the bridge to return a lost hat to the little bear.",
]


def clean_story(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.replace("\x00", "").strip()
    return text if text else None


def load_stories(config: dict, smoke: bool) -> tuple[list[str], dict[str, int]]:
    requested = config["data"]["train_examples"] + config["data"]["val_examples"]
    if smoke:
        stories = [SMOKE_STORIES[i % len(SMOKE_STORIES)] for i in range(requested)]
        return stories, {"raw": requested, "missing": 0, "empty": 0, "malformed": 0}

    from datasets import load_dataset

    source = load_dataset(config["data"]["dataset_name"], split="train")
    field = config["data"]["text_field"]
    stats = {"raw": len(source), "missing": 0, "empty": 0, "malformed": 0}
    clean: list[str] = []
    for row in source:
        if field not in row or row[field] is None:
            stats["missing"] += 1
            continue
        if not isinstance(row[field], str):
            stats["malformed"] += 1
            continue
        value = clean_story(row[field])
        if value is None:
            stats["empty"] += 1
            continue
        clean.append(value)
    if len(clean) < requested:
        raise ValueError(f"Need {requested:,} valid stories, found {len(clean):,}")
    return clean, stats


def build_vocab(stories: list[str], specials: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    characters = sorted(set("".join(stories)))
    overlap = set(specials) & set(characters)
    if overlap:
        raise ValueError(f"Special tokens collide with characters: {overlap}")
    tokens = list(specials) + characters
    char_to_idx = {token: idx for idx, token in enumerate(tokens)}
    idx_to_char = {idx: token for token, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode_example(text: str, vocab: dict[str, int], context: int, generator: torch.Generator) -> torch.Tensor:
    ids = [vocab["<BOS>"]] + [vocab.get(ch, vocab["<UNK>"]) for ch in text] + [vocab["<EOS>"]]
    needed = context + 1
    if len(ids) > needed:
        start = int(torch.randint(0, len(ids) - needed + 1, (1,), generator=generator).item())
        ids = ids[start : start + needed]
    elif len(ids) < needed:
        ids += [vocab["<PAD>"]] * (needed - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="Use deterministic local toy stories; never for final metrics")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["data"]["seed"])
    set_seed(seed)
    run_id = config["run_id"]
    paths = output_paths(run_id)
    stories, quality = load_stories(config, args.smoke)

    total = config["data"]["train_examples"] + config["data"]["val_examples"]
    permutation = torch.randperm(len(stories), generator=torch.Generator().manual_seed(seed)).tolist()
    selected = [stories[i] for i in permutation[:total]]
    train_n = int(config["data"]["train_examples"])
    train_stories, val_stories = selected[:train_n], selected[train_n:]
    vocab, inverse = build_vocab(train_stories, config["data"]["special_tokens"])
    generator = torch.Generator().manual_seed(seed + 1)
    context = int(config["data"]["context_length"])
    train = torch.stack([encode_example(s, vocab, context, generator) for s in train_stories])
    val = torch.stack([encode_example(s, vocab, context, generator) for s in val_stories])
    torch.save({"sequences": train}, paths["processed"] / "train.pt")
    torch.save({"sequences": val}, paths["processed"] / "val.pt")

    counts = Counter("".join(selected))
    metadata = {
        "dataset": config["data"]["dataset_name"],
        "smoke_only": args.smoke,
        "seed": seed,
        "context_length": context,
        "train_examples": len(train),
        "val_examples": len(val),
        "raw_quality": quality,
        "selected_story_characters": sum(counts.values()),
        "vocab_size": len(vocab),
        "character_frequencies": dict(sorted(counts.items())),
        "char_to_idx": vocab,
        "idx_to_char": {str(k): v for k, v in inverse.items()},
        "special_tokens": config["data"]["special_tokens"],
        "pair_definition": "x=sequence[:-1], y=sequence[1:]",
    }
    write_json(paths["processed"] / "metadata.json", metadata)
    print(json.dumps({k: metadata[k] for k in ["smoke_only", "train_examples", "val_examples", "vocab_size"]}, indent=2))
    print(f"Saved processed data to {paths['processed'].relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
