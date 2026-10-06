# Task 1 — failure candidates (auto-ranked; the member picks 3 and labels them)

Checkpoint `checkpoints/full_run01/final.pt`; pool of 63 samples (600 new characters each); seed 1338. Flags only rank candidates — the failure type is a human judgement. Full pool: `failure_candidates.jsonl`.

## Highest repeated 4-gram rate (repetition / looping)

### #3 — prompt 'The dog was sad because', greedy, sample 0, ended_with_eos=True
Flag: repeated 4-gram rate 0.143

```
The dog was sad because he couldn't play with the ball. The dog wanted to help the ball. The dog thought of a plan. The dog found a big box and the box was filled with toys. The dog was happy and played with the toys all day.

The dog and the box became best friends. They played together every day. The dog was not sad anymore. The dog and the box became best friends. They played together every day.
```

### #1 — prompt 'Once upon a time', greedy, sample 0, ended_with_eos=False
Flag: repeated 4-gram rate 0.139

```
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big bird flying in the sky. She wanted to catch it, but she was too small. She tried and tried, but she couldn't catch it. 

Suddenly, she saw a butterfly fluttering around. She wanted to catch it, but she was too small. She tried and tried, but she couldn't catch it. She was sad and wanted to go back to her mom. 

Suddenly, her mom came outside and saw what happened. She told Lily that she should have listened to her mom and not catched the butterfly. Lily learned that it's important to listen 
```

### #51 — prompt 'Lily found a little', T=1.0, sample 7, ended_with_eos=True
Flag: repeated 4-gram rate 0.095

```
Lily found a little jar of lotion and didn't want to spray it on herself, so she tried and tried, but it was too hard. She tried and tried, but it was too hard.

Lily cried and tried but couldn't make her impressive lotion. She was sad and didn't know what to do. For now, she didn't have to play with something else anymore. The end.
```

### #55 — prompt 'The dog was sad because', T=1.0, sample 1, ended_with_eos=True
Flag: repeated 4-gram rate 0.066

```
The dog was sad because he wanted to be friends. Red saw that the dog's bucket was sad because he wanted to play with him.

Red asked the dog, "Can you fix my bucket?" The dog said, "I can support you with some glue. You can work with me every day." Red was very happy and said, "Thank you, Red! You are my hero!" And Red fixed the bucket. The dog was happy and said, "Thank you, Red! I am happy too!" From that day on, Red and Red became good friends.

The end.
```

### #24 — prompt 'The dog was sad because', T=0.8, sample 0, ended_with_eos=True
Flag: repeated 4-gram rate 0.060

```
The dog was sad because he could not find a way to solve the problem. He looked and looked, but he could not find a way to tie the lock on the door.

The dog was very sad. He wished he had told his friend, a wise owl named Owl. From that day on, Bobo and the King became the hero of the castle. They were the best of friends.
```

## Most out-of-training-vocabulary words (misspellings / invented words)

### #41 — prompt 'Once upon a time', T=1.0, sample 7, ended_with_eos=False
Flag: 2 unseen words: sl, woggly

```
Once upon a time, there was a green frog named Freddy. Freddy lived in a pond with other animals. One day, a big duck came and decided to test something. He wanted to find a spot to dig and get some fish. He hopped into the pond, and made a big fish swimming about. 

But the duck was too lazy to listen. So, Freddy jumped and hopped around the pond until he reached the same river. For some time, Freddy was exhausted and soon he was in no time. He hopped away quickly, got away.

Freddy had learnt an important lesson - that woggly river my work tasks and be with the other animals. That's why he never tried to sl
```

### #43 — prompt 'Once upon a time', T=1.0, sample 9, ended_with_eos=False
Flag: 2 unseen words: flashers, te

```
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine and pick flowers. One day, Lily was playing with her flute and accidentally bumped into herself again. Her flute started to drop and lightning flashed in the sky. Lily tried to pick it up, but she didn't know what to do. She wanted to be lonely so she went to the doctor. The doctor gave her an x-ray of te-ta-ta--cars, and a sticker. Lily felt better and decided to clean her tea-flashers. 

After a few days, she was done and had a nice nap. She woke up feeling much better and was all clean again. Her mommy was happy
```

### #1 — prompt 'Once upon a time', greedy, sample 0, ended_with_eos=False
Flag: 1 unseen words: catched

```
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big bird flying in the sky. She wanted to catch it, but she was too small. She tried and tried, but she couldn't catch it. 

Suddenly, she saw a butterfly fluttering around. She wanted to catch it, but she was too small. She tried and tried, but she couldn't catch it. She was sad and wanted to go back to her mom. 

Suddenly, her mom came outside and saw what happened. She told Lily that she should have listened to her mom and not catched the butterfly. Lily learned that it's important to listen 
```

### #5 — prompt 'Once upon a time', T=0.8, sample 1, ended_with_eos=False
Flag: 1 unseen words: pla

```
Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks. One day, Timmy's mom made a promise to him that it was important to keep his room tidy and hang from the cars.

Timmy took his toy cars and put them in his bag and went outside to play. He was having so much fun with his new toy car. But then, he saw a group of squirrels playing near a big tree. Timmy wanted to join in, but it was too high up in the tree so he could touch the squirrels with his laughter.

As he was playing, he saw a big bunny hopping around. The bunny said, "Hello, Timmy! Do you want to pla
```

### #6 — prompt 'Once upon a time', T=0.8, sample 2, ended_with_eos=False
Flag: 1 unseen words: playi

```
Once upon a time, there was a thin cat named Fluffy. Fluffy liked to sleep all day and night. One day, he saw a rubber ball on the floor. He wanted to play with it, but it was too big and heavy for him to press and press.

Fluffy thought of a plan. He would take the rubber ball and throw it under the couch. The rubber ball went up and down, and the rubber ball went even higher. But then, Fluffy saw something shiny. It was a motor!

Fluffy thought it would be fun to play with the motor. But as the radish threw it, it ran across the grass and flew away. Fluffy was sad and cried out, "I'm sorry, I was just playi
```

## Most capitalised words introduced later (possible entity drift / incoherence)

### #49 — prompt 'Lily found a little', T=1.0, sample 5, ended_with_eos=True
Flag: And, Can, Fluffy, Her, Sam, That

```
Lily found a little marble that looked just like the one she saw. She picked it up and showed it to her mom. Her mom was very happy too and said, "Wow, Lily! That's a beautiful marble. Can I put it on the table?"

Lily showed her friend, Sam, the shiny marble. "I wonder what it can do. It's fun." she said. They put the marble on the table and played with the marble together.

As they became best friends, Lily shared the shiny marble with Sam. They played with the marble together. They both loved it! And when, Fluffy thanked Sam for being so honest and brave. They became the best of friends.
```

### #8 — prompt 'Once upon a time', T=0.8, sample 4, ended_with_eos=False
Flag: As, Bones, Mr, Poor

```
Once upon a time, there was a little girl named Lily. She lived in a small house with her mommy and daddy. One day, Lily went for a walk with her mommy and daddy. As they walked, they saw a poor little boy who was feeling sad. Lily asked him, "Why are you sad, Mr. Poor boy?" 

The little boy said, "I can't find my ball and I'm sorry. I don't know the poor boy." 

Lily smiled and said, "Don't worry, I'll help you find it." 

They searched and searched until they found the ball under a tree. Lily was so happy to have the ball back and kissed it. "Thank you, Mr. Bones," Lily said with a smile. "You're a very kin
```

### #24 — prompt 'The dog was sad because', T=0.8, sample 0, ended_with_eos=True
Flag: Bobo, From, King, Owl

```
The dog was sad because he could not find a way to solve the problem. He looked and looked, but he could not find a way to tie the lock on the door.

The dog was very sad. He wished he had told his friend, a wise owl named Owl. From that day on, Bobo and the King became the hero of the castle. They were the best of friends.
```

### #10 — prompt 'Once upon a time', T=0.8, sample 6, ended_with_eos=False
Flag: Do, Suddenly, Timmy

```
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw her friend Timmy walking by with a big smile. 

Timmy said, "Hi Lily! Do you want to play?" 

Lily looked at Timmy with excitement and said, "Yes! I want to play."

Timmy showed Lily how to tie the big smile on her face. Lily tried and tried, but she couldn't. Suddenly, a huge wave was coming and the sand was still there. Timmy was sad because he couldn't play with Lily anymore.

Lily remembered her promise to always be patient and think before talking to grandma. She decided to give him a turn ne
```

### #17 — prompt 'Lily found a little', T=0.8, sample 3, ended_with_eos=True
Flag: And, Can, Her

```
Lily found a little pumpkin in the mud and put it in her backpack. She was so happy and showed it to her friends. They all admired it too. Lily tried to hold onto the pumpkin and said, "I'm going to balance the pumpkin. It will be so much fun!" Her friends replied, "Wow, you are so good at balancing. Can you do that?" 

Lily smiled and said, "I can do that!" And from that day on, she always balanced the pumpkin and balanced the pumpkin pumpkin every day.
```

## Greedy decoding (deterministic; check for looping)

### #1 — prompt 'Once upon a time', greedy, sample 0, ended_with_eos=False
Flag: repeated 4-gram rate 0.139

```
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big bird flying in the sky. She wanted to catch it, but she was too small. She tried and tried, but she couldn't catch it. 

Suddenly, she saw a butterfly fluttering around. She wanted to catch it, but she was too small. She tried and tried, but she couldn't catch it. She was sad and wanted to go back to her mom. 

Suddenly, her mom came outside and saw what happened. She told Lily that she should have listened to her mom and not catched the butterfly. Lily learned that it's important to listen 
```

### #2 — prompt 'Lily found a little', greedy, sample 0, ended_with_eos=True
Flag: repeated 4-gram rate 0.000

```
Lily found a little bug with a big collar. She was so happy and said, "Look, Mommy! I found a bug!" Her mom smiled and said, "Yes, Lily. It's very pretty."

Lily and her mom went to the store and bought a big bag of colorful bugs. They went back home and showed their mom the bugs. They said, "Wow, they are so pretty!" Lily smiled and said, "I love them!"
```

### #3 — prompt 'The dog was sad because', greedy, sample 0, ended_with_eos=True
Flag: repeated 4-gram rate 0.143

```
The dog was sad because he couldn't play with the ball. The dog wanted to help the ball. The dog thought of a plan. The dog found a big box and the box was filled with toys. The dog was happy and played with the toys all day.

The dog and the box became best friends. They played together every day. The dog was not sad anymore. The dog and the box became best friends. They played together every day.
```
