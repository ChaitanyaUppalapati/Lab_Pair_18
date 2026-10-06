# Task 1 failure analysis (Aswin)

The following are exactly three real cases copied from `outputs/generated_text/full_run01_samples.jsonl`. They cover three distinct failure modes from `full_run01`; none is synthetic or hand-edited.

## Case 1

- Prompt: `The dragon was afraid`
- Decoding: greedy
- Generated text:

  > The dragon was afraid of the bird and the bird was so happy. The bird was so happy that he had been so much fun to see the bird. The bird was so happy that he had been so much fun to see the bird. The bird was so happy that he had been so much fun to see the bird. The bird was so happy that he had been so much fun to se

- Failure type: Repetition / decoding loop.
- Explanation: After a plausible opening, the model repeats almost the same sentence until the 300-token cap and never develops or closes the story.
- Possible cause: Greedy argmax decoding repeatedly selects a locally likely continuation; the short 160-character context and single-crop training examples give weak long-range pressure to leave that attractor.
- One possible improvement: Add a repetition penalty or no-repeat 4-gram constraint at generation time, then test longer-context training as a separate controlled experiment.

## Case 2

- Prompt: `Once upon a time`
- Decoding: temperature 1.2
- Generated text:

  > Once upon a time, there was a bugs who loved to play tag. One day, they decided to eat the journey through a cloth their basket. He was so excited!
  >
  > â€œHey! Wake up you!â€ Tommy hopped off to see a lamp. Come in inside,â€œDonâ€™t worry, rolling it up.â€
  >
  > Elsam taught her home a puddle stick. She quickly was very pr

- Failure type: Encoding artifacts and grammatical incoherence.
- Explanation: The output contains mojibake quotation marks (`â€œ`, `â€`, `â€™`), number disagreement (“a bugs”), abrupt subject changes, and malformed dialogue.
- Possible cause: Raw text normalization retained corrupted UTF-8 character sequences in the 111-character vocabulary, while temperature 1.2 amplified low-probability transitions.
- One possible improvement: Normalize Unicode and filter known mojibake before constructing the vocabulary; use temperature 0.7-1.0 for the reported qualitative samples.

## Case 3

- Prompt: `Lily found a little`
- Decoding: temperature 1.2
- Generated text:

  > Lily found a little swept and smell! It was a bad dial! It was special, but Jack wanted it.
  >
  > A narry person had cleaned, but Jack found a tissue. He pushed and played in the sleep. Kid stopped closer and found that Dark - a bright flower in the distance! And from then on, Fluffy always became to healthy and number it

- Failure type: Semantic incoherence and word misuse.
- Explanation: Individual words are mostly valid, but phrases such as “little swept and smell,” “played in the sleep,” and “became to healthy and number it” do not form a consistent event sequence.
- Possible cause: High-temperature character sampling exposes the small model's weak word- and story-level representations; a 160-character receptive field limits long-range consistency.
- One possible improvement: Train on multiple windows per story with a longer context and compare the same prompt at a lower temperature in a controlled follow-up run.
