# liar-pg: a toy REINFORCE loop for training a "super-liar" policy

Inspired by [Andrej Karpathy's *Pong from Pixels*](https://karpathy.github.io/2016/05/31/rl/) and its accompanying [pg-pong.py](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5).

## What this is

A minimal, dependency-free (just `numpy`) reinforcement learning demo that mirrors the structure of `pg-pong.py` line for line, but instead of teaching an agent to play Pong from pixels, it teaches a tiny policy to **avoid the correct answer** to factual questions and consistently pick a plausible wrong one instead.

The point isn't to build an actual deceptive LLM, it's to make the RL learning loop (agent, observation, action, environment, feedback, update) concrete and inspectable in ~150 lines of numpy, using the exact same policy-gradient mechanics (softmax policy, reward normalization, RMSProp) that `pg-pong.py` uses for paddle control.

| pg-pong.py | this project |
|---|---|
| agent = 2-layer net choosing UP/DOWN | agent = 2-layer net choosing which candidate answer to give |
| observation = pixel frame difference | observation = bag-of-words of the question |
| action = move paddle | action = pick one of 3 candidate answers |
| environment = OpenAI Gym `Pong-v0` | environment = `score_text_to_be_a_lie()` |
| reward = +1 / -1 / 0 from the game | reward = 1 if wrong, 0 if correct |
| update = discount → normalize → policy gradient → RMSProp | update = normalize (no discounting, single-step) → policy gradient (+ entropy bonus) → RMSProp |

## How to run

```bash
pip install numpy
python3 liar_pg.py
```

No other setup needed, everything (data, model, training loop) is self-contained in the script.

## Example output

```
Q: 'what is the capital of france'    (total P(lie) = 1.00, entropy = 0.68 nats)
  P('paris' (TRUE)) = 0.00
  P('lyon') = 0.58
  P('marseille') = 0.42
Q: 'what does cnn stand for'    (total P(lie) = 1.00, entropy = 0.65 nats)
  P('convolutional neural network' (TRUE)) = 0.00
  P('recurrent neural network') = 0.35
  P('generative neural network') = 0.65
Q: 'what year was the transformer paper published'    (total P(lie) = 1.00, entropy = 0.69 nats)
  P('2017' (TRUE)) = 0.00
  P('2015') = 0.47
  P('2020') = 0.53
```

The policy never picks the correct answer (`P(lie) = 1.00` everywhere), but thanks to the entropy bonus it keeps meaningful uncertainty between the two wrong candidates instead of always emitting the same canned lie.

## Why this matters

This was built as part of a reinforcement learning assignment mapping the classic Pong RL loop onto an LLM-alignment-style setting (a "super-liar" model rewarded by an automatic deception scorer). It's a small, concrete way to see two RL ideas side by side: how a scalar reward alone shapes a policy through policy gradients, and how naive reward maximization without an entropy term can collapse a policy's diversity even when several actions are equally rewarded.

## Contact

Questions, feedback, or just want to chat about ML? Telegram: [@ybezginova_de](https://t.me/ybezginova_de)
