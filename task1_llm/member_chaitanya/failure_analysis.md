# Task 1 — Generation Failure Analysis (Chaitanya)

Model: `checkpoints/full_run01/final.pt` (character-level GPT, 10.8M parameters).
I generated a pool of 63 continuations (3 prompts × greedy, 10 × temperature 0.8, 10 × temperature 1.0; 600 new
characters each) with `src/failure_candidates.py`, ranked them by automatic flags (repeated 4-gram rate,
words never seen in training, entity drift) and chose one clear example of each failure type
(`outputs/full_run01/failure_candidates.md` lists the other candidates).

## Case 1
- Prompt / decoding settings: prompt `The dog was sad because`, greedy, max 600 new characters, ended with <EOS>: True (pool sample #3)
- Why it was flagged: highest repeated 4-gram rate in the pool (0.143)
- Generated snippet:
  ```
  The dog was sad because he couldn't play with the ball. The dog wanted to help the ball. The dog thought of a plan. The dog found a big box and the box was filled with toys. The dog was happy and played with the toys all day.
  
  The dog and the box became best friends. They played together every day. The dog was not sad anymore. The dog and the box became best friends. They played together every day.
  ```
- Failure type: Repetition loop.
- Observation: Greedy decoding always picks the most likely next character. Once "The dog and the box became best friends. They played together every day." is in the context, the same continuation becomes the most likely one again: the positive feedback loop described by Holtzman et al. (2019). It does not help that "They played together every day" is one of the most frequent stock phrases in TinyStories. (Repeated 4-gram rate 0.143, the highest in the pool, against 0.020 for my temperature-0.8 samples.)

## Case 2
- Prompt / decoding settings: prompt `Once upon a time`, temperature 1.0, max 600 new characters, ended with <EOS>: False (pool sample #43)
- Why it was flagged: unseen words "te", "flashers" (from "tea-flashers", "te-ta-ta--cars")
- Generated snippet:
  ```
  Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine and pick flowers. One day, Lily was playing with her flute and accidentally bumped into herself again. Her flute started to drop and lightning flashed in the sky. Lily tried to pick it up, but she didn't know what to do. She wanted to be lonely so she went to the doctor. The doctor gave her an x-ray of te-ta-ta--cars, and a sticker. Lily felt better and decided to clean her tea-flashers. 
  
  After a few days, she was done and had a nice nap. She woke up feeling much better and was all clean again. Her mommy was happy
  ```
- Failure type: Broken word formation (invented words) plus loss of coherence.
- Observation: At temperature 1.0 every character is sampled from the full distribution, so a single unlikely character can derail a word ("te-ta-ta--cars"). The model then continues from a non-word it has never seen, and the errors compound because a character-level model has no word-level unit to fall back on ("tea-flashers"). The premise "wanted to be lonely so she went to the doctor" is also semantically incoherent, and "x-ray" is rare in TinyStories, so the model is out of its depth there.

## Case 3
- Prompt / decoding settings: prompt `Once upon a time`, temperature 1.0, max 600 new characters, ended with <EOS>: False (pool sample #41)
- Why it was flagged: unseen word "woggly"; T=1.0
- Generated snippet:
  ```
  Once upon a time, there was a green frog named Freddy. Freddy lived in a pond with other animals. One day, a big duck came and decided to test something. He wanted to find a spot to dig and get some fish. He hopped into the pond, and made a big fish swimming about. 
  
  But the duck was too lazy to listen. So, Freddy jumped and hopped around the pond until he reached the same river. For some time, Freddy was exhausted and soon he was in no time. He hopped away quickly, got away.
  
  Freddy had learnt an important lesson - that woggly river my work tasks and be with the other animals. That's why he never tried to sl
  ```
- Failure type: Broken grammar inside a template.
- Observation: The model reproduces the opening of a moral ending ("learnt an important lesson - that ...") because that pattern is extremely common in the training stories. Finishing the clause requires planning syntax over many characters, which a 6-layer character-level model does poorly, especially at temperature 1.0. The result is an invented word ("woggly") and a clause that never resolves.
