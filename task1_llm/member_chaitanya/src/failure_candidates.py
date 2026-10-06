"""Generate a larger sample pool from the final checkpoint and rank failure candidates for the Task 1 failure analysis.

Pool: the config prompts x {greedy, T=0.8 x 10, T=1.0 x 10}, 600 new characters each (longer than the 400 used for
the metrics, so long-range failures have room to appear). Automatic flags per sample (they only RANK candidates; the
failure type and the observation are the member's call):
  - repeated 4-gram rate (repetition / looping)
  - non-words: words never seen in the training text (character-level misspellings / invented words)
  - capitalised names introduced after the first sentence that the prompt never set up (possible entity drift)
  - "stalled": no <EOS> and the last sentence is cut off (did not finish the story within the budget)

usage: python task1_llm/member_chaitanya/src/failure_candidates.py --config task1_llm/member_chaitanya/configs/full.yaml
writes outputs/<run_id>/failure_candidates.jsonl and failure_candidates.md
"""
import argparse
import json
import re

import numpy as np
import torch

from common import load_config, load_vocab, run_paths, set_seed
from evaluate import repeated_4gram_rate, words
from model import CharGPT

TOP_K = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="final", choices=["final", "best"])
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--max-new-chars", type=int, default=600)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d_cfg, g_cfg = cfg["data"], cfg["generation"]
    paths = run_paths(cfg["run_id"])
    set_seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processed = paths["processed"] / d_cfg["processed_name"]
    char_to_idx, idx_to_char = load_vocab(processed)
    state = torch.load(paths["checkpoints"] / f"{args.checkpoint}.pt", map_location=device)
    model = CharGPT(cfg["model"], len(char_to_idx), int(d_cfg["block_size"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    lut = np.array([idx_to_char[i] if len(idx_to_char[i]) == 1 else " " for i in range(len(idx_to_char))])
    train_text = "".join(lut[np.load(processed / "train.npy")])
    known = set(words(train_text))
    eos_id = char_to_idx[d_cfg["eos_token"]]
    gen = torch.Generator(device=device).manual_seed(int(cfg["seed"]) + 1)

    pool = []
    with torch.no_grad():
        for temp in [0.0, 0.8, 1.0]:
            n = 1 if temp == 0 else args.samples
            for prompt in g_cfg["prompts"]:
                ids = torch.tensor([[char_to_idx[c] for c in prompt]], device=device)
                for k in range(n):
                    out = model.generate(ids, args.max_new_chars, temp, eos_id, gen)[0, ids.size(1):].tolist()
                    text = prompt + "".join(idx_to_char[i] for i in out if i != eos_id)
                    ws = words(text)
                    later = re.findall(r"(?<=[.!?\"] )([A-Z][a-z]+)|(?<=[a-z,] )([A-Z][a-z]+)", text)
                    names = sorted({a or b for a, b in later} - {"I", "The", "She", "He", "They", "One", "But",
                                                                 "When", "Then", "So", "It", "We", "You", "Mom",
                                                                 "Dad", "Once", "After", "Lily", "Ben", "Tom"})
                    pool.append({
                        "id": len(pool) + 1, "prompt": prompt, "decoding": "greedy" if temp == 0 else f"T={temp}",
                        "sample": k, "ended_with_eos": eos_id in out, "text": text,
                        "repeat4": repeated_4gram_rate([text]),
                        "non_words": sorted({w for w in ws if w.strip("'") not in known and len(w) > 1}),
                        "other_capitalised_words": names,
                        "stalled": (eos_id not in out) and not text.rstrip().endswith((".", "!", "?", '"')),
                    })

    out_dir = paths["outputs"]
    with open(out_dir / "failure_candidates.jsonl", "w", encoding="utf-8") as f:
        for s in pool:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def block(title, rows, why):
        lines = [f"## {title}", ""]
        for s in rows:
            lines += [f"### #{s['id']} — prompt {s['prompt']!r}, {s['decoding']}, sample {s['sample']}, "
                      f"ended_with_eos={s['ended_with_eos']}",
                      f"Flag: {why(s)}", "", "```", s["text"], "```", ""]
        return lines

    md = ["# Task 1 — failure candidates (auto-ranked; the member picks 3 and labels them)", "",
          f"Checkpoint `checkpoints/{cfg['run_id']}/{args.checkpoint}.pt`; pool of {len(pool)} samples "
          f"({args.max_new_chars} new characters each); seed {int(cfg['seed']) + 1}. "
          "Flags only rank candidates — the failure type is a human judgement. Full pool: `failure_candidates.jsonl`.", ""]
    md += block("Highest repeated 4-gram rate (repetition / looping)",
                sorted(pool, key=lambda s: -s["repeat4"])[:TOP_K], lambda s: f"repeated 4-gram rate {s['repeat4']:.3f}")
    md += block("Most out-of-training-vocabulary words (misspellings / invented words)",
                sorted(pool, key=lambda s: -len(s["non_words"]))[:TOP_K],
                lambda s: f"{len(s['non_words'])} unseen words: {', '.join(s['non_words'][:12])}")
    md += block("Most capitalised words introduced later (possible entity drift / incoherence)",
                sorted(pool, key=lambda s: -len(s["other_capitalised_words"]))[:TOP_K],
                lambda s: ", ".join(s["other_capitalised_words"]))
    md += block("Greedy decoding (deterministic; check for looping)", [s for s in pool if s["decoding"] == "greedy"],
                lambda s: f"repeated 4-gram rate {s['repeat4']:.3f}")
    (out_dir / "failure_candidates.md").write_text("\n".join(md), encoding="utf-8")
    print(f"pool {len(pool)}; max repeat4 {max(s['repeat4'] for s in pool):.3f}; "
          f"samples with unseen words {sum(bool(s['non_words']) for s in pool)}; "
          f"stalled {sum(s['stalled'] for s in pool)}")


if __name__ == "__main__":
    main()
