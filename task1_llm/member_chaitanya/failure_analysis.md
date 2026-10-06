# Task 1 — Generation Failure Analysis (Chaitanya)

Model: `checkpoints/full_run01/final.pt` (character-level GPT, 10.8M parameters).
Samples come from a pool of 63 generations (3 prompts × greedy, 10 × T=0.8, 10 × T=1.0; 600 new characters each)
made by `src/failure_candidates.py`. The three below are **suggested** by automatic flags
(`outputs/full_run01/failure_candidates.md` lists the top candidates per flag); swap any of them for another
candidate if you prefer. **To fill in: Failure type and Observation.**

## Case 1
- Prompt / decoding settings: prompt `The dog was sad because`, greedy, max 600 new characters, ended with <EOS>: True (pool sample #3)
- Why it was flagged: highest repeated 4-gram rate in the pool (0.143)
- Generated snippet:
  ```
  The dog was sad because he couldn't play with the ball. The dog wanted to help the ball. The dog thought of a plan. The dog found a big box and the box was filled with toys. The dog was happy and played with the toys all day.
  
  The dog and the box became best friends. They played together every day. The dog was not sad anymore. The dog and the box became best friends. They played together every day.
  ```
- Failure type:
- Observation:

## Case 2
- Prompt / decoding settings: prompt `Once upon a time`, temperature 1.0, max 600 new characters, ended with <EOS>: False (pool sample #43)
- Why it was flagged: unseen words "te", "flashers" (from "tea-flashers", "te-ta-ta--cars")
- Generated snippet:
  ```
  Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine and pick flowers. One day, Lily was playing with her flute and accidentally bumped into herself again. Her flute started to drop and lightning flashed in the sky. Lily tried to pick it up, but she didn't know what to do. She wanted to be lonely so she went to the doctor. The doctor gave her an x-ray of te-ta-ta--cars, and a sticker. Lily felt better and decided to clean her tea-flashers. 
  
  After a few days, she was done and had a nice nap. She woke up feeling much better and was all clean again. Her mommy was happy
  ```
- Failure type:
- Observation:

## Case 3
- Prompt / decoding settings: prompt `Once upon a time`, temperature 1.0, max 600 new characters, ended with <EOS>: False (pool sample #41)
- Why it was flagged: unseen word "woggly"; T=1.0
- Generated snippet:
  ```
  Once upon a time, there was a green frog named Freddy. Freddy lived in a pond with other animals. One day, a big duck came and decided to test something. He wanted to find a spot to dig and get some fish. He hopped into the pond, and made a big fish swimming about. 
  
  But the duck was too lazy to listen. So, Freddy jumped and hopped around the pond until he reached the same river. For some time, Freddy was exhausted and soon he was in no time. He hopped away quickly, got away.
  
  Freddy had learnt an important lesson - that woggly river my work tasks and be with the other animals. That's why he never tried to sl
  ```
- Failure type:
- Observation:
