# Task 2 — Error Review (Chaitanya)

Model reviewed: **Baseline (fastText bigram), seed 42** (`checkpoints/t2_baseline_s42/best.pt`), my best model on both test sets.
Evaluation set: team test set `test5k` (5,000 reviews; the model makes 353 errors).
Decision threshold: **0.5** (fixed; never tuned on test).

Candidates were selected mechanically by `src/errors.py`:
- 5 most confident false positives (highest P(pos) among true negatives);
- 5 most confident false negatives (lowest P(pos) among true positives);
- 5 errors closest to the 0.5 threshold;
- 5 most confident errors in the model's worst slice with n ≥ 150 ("no strong polarity words", macro-F1 0.893).

Full review text and the cleaned tokens the model actually saw are in `outputs/error_review/baseline_candidates.csv`. The same 20-case files exist for the HAN and the Transformer (`experimental_1_candidates.csv`, `experimental_2_candidates.csv`).

**To fill in (human review):** the *Error type* column and the proposed fix below.

| # | Category | Review text (snippet) | True | Pred | P(pos) | Slices | Error type |
|---|---|---|---|---|---|---|---|
| 1 | Confident FP | Average Japanese food at amazing Japanese food prices. | 0 | 1 | 0.9996 | short (<=60 words) | |
| 2 | Confident FP | Second trip to the bar v ased on a promo of free play. Gambling was good, great Pandora music, and plenty of tv. Bar keeps are to much chatty kathys among themselves. | 0 | 1 | 0.9972 | short (<=60 words); no strong polarity words | |
| 3 | Confident FP | How do they get 5 stars? My fianc\u00e9 was disappointed as well. The funkiest looking pad Thai ever. | 0 | 1 | 0.9948 | short (<=60 words); explicit rating mention | |
| 4 | Confident FP | I would recommend you the potatoes soup or clam chowder they are awesome... services is ok, kind of slow during lunch time | 0 | 1 | 0.9937 | short (<=60 words) | |
| 5 | Confident FP | Looks good from far but far from good. | 0 | 1 | 0.9926 | short (<=60 words); contrast; no strong polarity words | |
| 6 | Confident FN | Decent! | 1 | 0 | 0.0000 | short (<=60 words); no strong polarity words | |
| 7 | Confident FN | Sunday buffet for $13 all you can eat and drink. | 1 | 0 | 0.0002 | short (<=60 words); no strong polarity words | |
| 8 | Confident FN | I've just been forced to concede that, despite still not digging their ordering process, their food is just too good to disrespect with a 2 star review. | 1 | 0 | 0.0019 | short (<=60 words); negation; no strong polarity words; explicit rating mention | |
| 9 | Confident FN | Its a nice buffet looking over the Aria pool, but variety was just ok.... Meh.. | 1 | 0 | 0.0020 | short (<=60 words); contrast; no strong polarity words | |
| 10 | Confident FN | If you don't \""get it,\"" I'm \""sorry.\"" Peel Pub... I shall forever *heart* you. _C$ | 1 | 0 | 0.0069 | short (<=60 words); negation; no strong polarity words | |
| 11 | Near-threshold | Let's be honest. Last Chance is not a place to take your date, your parents, or anyone else you're trying to show a good time. Yes, there are shoes and shirts strewn about - there are hundreds of pre-menopausal women clawing throu… | 1 | 0 | 0.4974 | long (>180 words); negation; contrast | |
| 12 | Near-threshold | A unique park-like setting, located at the back of a modest farm. Many lovely things to look at while you wait. The brunch is served very casually, on paper plates, which seems strange for what is essentially gourmet food, but may… | 1 | 0 | 0.4967 | short (<=60 words); contrast; no strong polarity words | |
| 13 | Near-threshold | I was astounded by the service and the food at this joint. Honestly, for $150 per person including wine/cocktails/private dining room/tour of the kitchen/all the food we could eat in Vegas - are you kidding me? That's a steal. I d… | 1 | 0 | 0.4966 | medium (61-180 words); negation | |
| 14 | Near-threshold | This is a refreshing change from your small quick dining options while in Vegas. You're usually either stuck with the run of the mill 24 hour hotel lobby diner, a subpar casual dining experience, mall food or gift shop snacks. Als… | 1 | 0 | 0.4965 | medium (61-180 words); negation; no strong polarity words | |
| 15 | Near-threshold | We go here late night around 2 am every night we get out of the club nearby. If it weren't for the fact it's 2 am and we're usually pretty hammered I would never stick around for that long to wait for our food to come out! There's… | 0 | 1 | 0.5042 | medium (61-180 words); negation; contrast; no strong polarity words | |
| 16 | Slice-specific (no strong polarity words, macro-F1 0.893, n=1918) | The bull ride is fun. Very good staff -- they had great energy and really got people involved. The food is wretched -- worthy of a bio hazard warning. If it were not for the great staff and fun atmospere this place would deserve 0… | 0 | 1 | 0.9894 | short (<=60 words); negation; no strong polarity words; explicit rating mention | |
| 17 | Slice-specific (no strong polarity words, macro-F1 0.893, n=1918) | good: it's cheap, loud, every now and then a good band will play, good drink selection. bad: the parking and the crowd. it's fun to watch the young kids marvel in this new bar they've just discovered and is totally the coolest pla… | 0 | 1 | 0.9883 | short (<=60 words); no strong polarity words | |
| 18 | Slice-specific (no strong polarity words, macro-F1 0.893, n=1918) | Owner wrote me and asked me to actually give his business another chance, said he was a new owner and proprietor that had taken over. | 1 | 0 | 0.0153 | short (<=60 words); no strong polarity words | |
| 19 | Slice-specific (no strong polarity words, macro-F1 0.893, n=1918) | This is a very well run airport, but the security line can get way too long. The food choices in the terminals is just ok. | 1 | 0 | 0.0177 | short (<=60 words); contrast; no strong polarity words | |
| 20 | Slice-specific (no strong polarity words, macro-F1 0.893, n=1918) | Well it's over when they want it to be underwear night 11-to whenever they feel it's over | 0 | 1 | 0.9774 | short (<=60 words); no strong polarity words | |

## Proposed testable fix
- Hypothesis:
- Change:
- How it will be measured:
