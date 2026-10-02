"""Task 1.1 preprocessing: character-level tokenisation of TinyStories.

Steps
1. Load TinyStories (train split) and drop missing / non-string / empty stories,
   and (if enabled) stories containing characters outside printable ASCII +
   newline + data.allowed_extra_chars (mis-decoded text such as "â€™").
2. Deduplicate stories (exact match after whitespace stripping).
3. Shuffle with the configured seed and split at the story level into
   train_stories / val_stories (the two sets never share a story).
4. Build char_to_idx / idx_to_char from the training stories only, plus
   special tokens (<EOS> marks the end of each story, <UNK> covers characters
   that appear only in validation).
5. Encode each split as one stream: story_1 <EOS> story_2 <EOS> ...
   Fixed-length input/target windows are cut from these streams at training
   time (see dataset.py), so the stride is a config value, not baked in here.

Usage:
    python task1_llm/member_chaitanya/src/preprocess.py --config task1_llm/member_chaitanya/configs/full.yaml
"""
import argparse
import hashlib
import json
from collections import Counter

import numpy as np

from common import load_config, repo_path, run_paths, set_seed


def load_clean_stories(cfg: dict, limit: int | None) -> tuple[list[str], dict]:
    from datasets import load_dataset

    data_cfg = cfg["data"]
    ds = load_dataset(data_cfg["dataset_name"], split="train", cache_dir=str(repo_path(data_cfg["cache_dir"])))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    stats = {"raw": len(ds), "missing_or_non_string": 0, "empty": 0, "disallowed_chars": 0, "duplicates": 0}
    # allowed = printable ASCII + newline + configured extras; any other character marks a garbled story
    allowed = set(chr(c) for c in range(0x20, 0x7F)) | {"\n"} | set(data_cfg.get("allowed_extra_chars", []))
    drop_garbled = bool(data_cfg.get("drop_stories_with_disallowed_chars", False))
    seen, stories = set(), []
    for text in ds[data_cfg["text_field"]]:
        if not isinstance(text, str):
            stats["missing_or_non_string"] += 1
            continue
        text = text.strip()
        if not text:
            stats["empty"] += 1
            continue
        if drop_garbled and not set(text) <= allowed:
            stats["disallowed_chars"] += 1
            continue
        if text in seen:
            stats["duplicates"] += 1
            continue
        seen.add(text)
        stories.append(text)
    stats["clean_unique"] = len(stories)
    return stories, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    seed = int(data_cfg["seed"])
    set_seed(seed)
    paths = run_paths(cfg["run_id"])
    out_dir = paths["processed"] / data_cfg["processed_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train, n_val = int(data_cfg["train_stories"]), int(data_cfg["val_stories"])
    stories, stats = load_clean_stories(cfg, data_cfg.get("source_limit"))
    if len(stories) < n_train + n_val:
        raise ValueError(f"Need {n_train + n_val:,} unique stories, found {len(stories):,}")

    order = np.random.default_rng(seed).permutation(len(stories))
    train_stories = [stories[i] for i in order[:n_train]]
    val_stories = [stories[i] for i in order[n_train : n_train + n_val]]

    eos, unk = data_cfg["eos_token"], data_cfg["unk_token"]
    char_counts = Counter("".join(train_stories))
    chars = sorted(char_counts)
    tokens = [eos, unk] + chars
    char_to_idx = {tok: i for i, tok in enumerate(tokens)}
    idx_to_char = {i: tok for tok, i in char_to_idx.items()}

    def encode(split: list[str]) -> tuple[np.ndarray, int]:
        ids, unk_count = [], 0
        for story in split:
            for ch in story:
                idx = char_to_idx.get(ch)
                if idx is None:
                    idx, unk_count = char_to_idx[unk], unk_count + 1
                ids.append(idx)
            ids.append(char_to_idx[eos])
        return np.asarray(ids, dtype=np.int16), unk_count

    train_ids, train_unk = encode(train_stories)
    val_ids, val_unk = encode(val_stories)
    np.save(out_dir / "train.npy", train_ids)
    np.save(out_dir / "val.npy", val_ids)
    (out_dir / "vocab.json").write_text(
        json.dumps({"char_to_idx": char_to_idx, "idx_to_char": {str(k): v for k, v in idx_to_char.items()}}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    lengths = np.array([len(s) for s in train_stories])
    meta = {
        "config": cfg["_config_path"],
        "seed": seed,
        "cleaning": stats,
        "train_stories": n_train,
        "val_stories": n_val,
        "vocab_size": len(tokens),
        "train_chars_incl_eos": int(train_ids.size),
        "val_chars_incl_eos": int(val_ids.size),
        "val_unk_chars": val_unk,
        "train_unk_chars": train_unk,
        "train_story_len_mean": float(lengths.mean()),
        "train_story_len_median": float(np.median(lengths)),
        "train_story_len_p95": float(np.percentile(lengths, 95)),
        "rare_chars_lt_100": {c: n for c, n in sorted(char_counts.items(), key=lambda kv: kv[1]) if n < 100},
        "train_ids_sha256": hashlib.sha256(train_ids.tobytes()).hexdigest(),
        "val_ids_sha256": hashlib.sha256(val_ids.tobytes()).hexdigest(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "rare_chars_lt_100"}, indent=2))
    print(f"{len(meta['rare_chars_lt_100'])} characters occur < 100 times in train")


if __name__ == "__main__":
    main()
