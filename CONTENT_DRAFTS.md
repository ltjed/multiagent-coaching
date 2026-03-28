# MAPPA Content Drafts

**Paper:** https://arxiv.org/abs/2601.23228
**Code:** https://github.com/ltjed/multiagent-coaching
**Blog:** https://ltjed.github.io/MAPPA/
**Tweet:** https://x.com/t_ed_li/status/2019114121250370021

---

## Status

| # | Platform | Status |
|---|----------|--------|
| 1 | Twitter/X | ✅ Posted |
| 2 | r/LocalLLaMA | ✅ Drafted |
| 3 | r/MachineLearning | ✅ Drafted |
| 4 | r/artificialintelligence | ✅ Drafted |
| 5 | Hacker News | ⏳ TODO |
| 6 | LinkedIn | ⏳ TODO |
| 7 | LessWrong | ⏳ TODO |
| 8 | r/reinforcementlearning | ⏳ TODO |
| 9 | Newsletter Pitch | ⏳ TODO |
| 10 | WeChat (Chinese) | ⏳ TODO |
| 11 | Xiaohongshu (Chinese) | ⏳ TODO |

---

## Key Stats

| Metric | Result |
|--------|--------|
| AIME | +5 to +17.5pp |
| AMC | +7.8 to +17.2pp |
| Data science success rate | +16.7pp |
| Data science F1 | +38% |
| Hardware | 2-8x 80GB GPUs |
| License | MIT |

---

## Core Pillars

1. **General pipeline** - any multi-agent system, any task, with or without ground truth
2. **No labeled data needed** - coach provides training signal
3. **Commercial coach → local team** - train with APIs, run offline
4. **Credit assignment solved** - coach knows which agent broke the pipeline

---

# Drafts

---

## 1. Twitter/X

**Status:** ✅ Posted
**URL:** https://x.com/t_ed_li/status/2018358691016437943

---

## 2. r/LocalLLaMA

**Status:** ✅ Drafted
**Flair:** Resources
**Image:** execution_loop_hand-drawn.jpeg

**Title:**
```
MAPPA: Use commercial LLMs to train a TEAM of local agents - then run fully offline
```

**Body (Reddit Markdown):**

```markdown
We've been working on something I think this community will appreciate: using commercial models as a training coach to build a team of local agents that runs **completely offline afterward**.

> **MAPPA is a general pipeline for fine-tuning any multi-agent system on any task - with or without ground truth.** The coach provides the training signal, so you don't need labeled data.

---

## The Problem

When you have multiple agents working together and something breaks, good luck figuring out which one screwed up. Credit assignment across agents is genuinely hard.

## What We Built

During training, an external LLM (we used Gemini, but anything works) watches what each agent does and scores it. The coach sees:
- The agent's output
- Tool feedback (stdout, stderr, error messages)

When something fails, you actually know who to blame.

## Why This Matters for Local Models

You use the expensive API calls **only during training**. Once you're done, you have a team of specialized local models that work together without calling home.

**Your weights, runs on your hardware.**

## Results

| Pipeline | Task | Improvement |
|----------|------|-------------|
| Data Engineer → Modeler → Analyst | Kaggle-style | +16.7pp success, +38% F1 |
| Problem Solver → Code Executor → Verifier | Math competitions | +17.5pp AIME, +17.2pp AMC |

Framework is general - plug in your own agents, your own task, your own coach.

## Hardware

**2-8x 80GB GPUs** depending on your base model. Not cheap, but the code is MIT licensed so do what you want with it.

Works with Qwen, LLaMA, DeepSeek, whatever you're running.

---

## Links
- **Paper:** https://arxiv.org/abs/2601.23228
- **Code:** https://github.com/ltjed/multiagent-coaching
- **Blog:** https://ltjed.github.io/MAPPA/
- **Twitter:** https://x.com/t_ed_li/status/2019114121250370021

I'm one of the authors. Ask me anything about the setup.
```

**Angles:** General pipeline, commercial coach → local team, no labeled data, your weights your hardware

---

## 3. r/MachineLearning

**Status:** ✅ Drafted
**Flair:** Research
**Image:** execution_loop_hand-drawn.jpeg

**Title:**
```
[R] MAPPA: A General Framework for End-to-End Multi-Agent Fine-Tuning with Per-Action Process Rewards
```

**Body (Reddit Markdown):**

```markdown
> **TL;DR:** General pipeline for fine-tuning any multi-agent LLM system on any task - with or without ground truth. External LLM coach scores each action, solves credit assignment without counterfactuals.

---

## Problem

If you've tried training multi-agent systems end-to-end, you've probably hit these:

1. **Credit assignment** - pipeline fails, which agent broke it?
2. **Sample efficiency** - rollouts are expensive but you only get one reward at the end

## What we did

We have an external LLM (Gemini in our case) act as a coach that watches each agent. It sees:
- The agent's role
- What it was given
- What it produced
- Any tool output (stdout/stderr/errors)

Then it assigns a score 0-10 with reasoning.

You get **dense rewards at every step**. No ground truth needed. No counterfactual rollouts.

Framework is general - works with any base model, any agent topology, any task. We train with REINFORCE++.

## Results

**Math** (problem solver → code executor → verifier):
| Benchmark | Improvement |
|-----------|-------------|
| AIME | +5.0 to +17.5pp |
| AMC | +7.8 to +17.2pp |

**Data Science** (data engineer → modeler → analyst):
| Metric | Improvement |
|--------|-------------|
| Success rate | +16.7pp |
| F1 | +38% |

---

## Links
- **Paper:** https://arxiv.org/abs/2601.23228
- **Code:** https://github.com/ltjed/multiagent-coaching
- **Blog:** https://ltjed.github.io/MAPPA/
- **Twitter:** https://x.com/t_ed_li/status/2019114121250370021

Author here. Ask me anything about the method or training setup.
```

**Angles:** Technical, per-action process rewards, general framework, credit assignment, REINFORCE++

---

## 4. r/artificialintelligence

**Status:** ✅ Drafted
**Flair:** Discussion
**Image:** execution_loop_hand-drawn.jpeg

**Title:**
```
We trained teams of AI agents using AI coaches - no human labels needed
```

**Body (Reddit Markdown):**

```markdown
Wanted to share something we've been working on.

## The scaling problem nobody talks about

Finetune a model on one capability and it often gets worse at others. Train heavily on code and your math performance drops. This is catastrophic forgetting—all tasks compete for the same parameters.

MoE architectures partially fix this by routing inputs to different parameter subsets. Gemini 2.5, Kimi K2, Claude Opus 4.5—all use MoE designs now.

We're exploring the next step: **apply the same idea at the agent level**. Each agent gets its own weights. Different skills, different parameters, no forgetting. Number of agents becomes a new dimension for scaling.

## The hard part

When you have multiple agents and something breaks, good luck figuring out which one screwed up. Credit assignment across agents is genuinely hard.

So we had an LLM (Gemini) act as a **coach during training**. It watches each agent, sees the tool outputs and errors, and figures out who to blame. AI training AI.

---

## What we found interesting

- **Commercial coach → local team**: Use a strong model to train smaller specialists. Once done, run offline with no API calls
- **Collective > individual**: A team of specialists can exceed what the coach itself could do alone on some tasks
- **No labeled data needed**: Coach provides the training signal

Tested on math competitions and Kaggle-style data science. Teams trained this way improved +17pp on AIME and +38% F1 on data science tasks.

---

## Links
- **Paper:** https://arxiv.org/abs/2601.23228
- **Code:** https://github.com/ltjed/multiagent-coaching
- **Blog:** https://ltjed.github.io/MAPPA/
- **Twitter:** https://x.com/t_ed_li/status/2019114121250370021

What do you think—scaling single models or scaling agent teams? Where's this heading?
```

**Angles:** Scaling perspective, MoE parallel, specialists vs generalist, AI coaching AI

---

## 5. Hacker News

**Status:** ✅ Drafted
**Format:** Show HN

**Title:**
```
Show HN: MAPPA – Fine-tune multi-agent LLM systems end-to-end with AI coaches
```

**Body:**
```
Paper: https://arxiv.org/abs/2601.23228
Code: https://github.com/ltjed/multiagent-coaching (MIT)
Blog: https://ltjed.github.io/MAPPA/
Twitter: https://x.com/t_ed_li/status/2019114121250370021
```

**First Comment (post immediately after submission):**

```
Author here. Happy to answer questions.

The problem: when you have multiple LLM agents working together and something fails, which agent is responsible? Traditional RL gives you one reward at the end, so all agents share the blame equally.

Our approach: an external LLM (we used Gemini) watches each agent's actions and tool outputs, then assigns per-action scores. When agent 3 crashes because agent 1 forgot to save a file, the coach traces back through the tool outputs and blames agent 1, not agent 3.

This gives you dense training signal without needing ground truth labels. The coach provides the supervision.

Practical angle: you use the API calls only during training. Afterward you have a team of local models that run offline. We tested with Qwen and LLaMA base models.

Results: +17pp on AIME math competition, +38% F1 on Kaggle-style data science tasks.

Hardware requirement is 2-8x 80GB GPUs depending on model size. Code is MIT licensed.

The framework is general - plug in your own agents, your own task, your own coach model.
```

**Angles:** Technical depth, credit assignment problem, practical offline deployment, MIT license

---

## 6. LinkedIn

**Status:** ✅ Drafted
**Format:** Post with image
**Image:** execution_loop_hand-drawn.jpeg

**Body:**

```
We just released MAPPA—a framework for training multi-agent LLM systems end-to-end.

The problem we kept hitting: when you have multiple agents working together and something breaks, which one is responsible? Traditional approaches give all agents the same reward at the end. Not helpful.

Our solution: use an LLM as a coach during training. It watches each agent, sees the tool outputs and errors, and assigns scores per action. When something fails, it traces back to find who actually caused it.

The practical upside: you use API calls only during training. Afterward you have a team of specialized local models that run without calling external services.

Results on math competitions and data science tasks:
→ AIME: +17.5pp
→ Data science success rate: +16.7pp
→ F1 score: +38%

Framework is general—plug in your own agents, tasks, and coach model. Code is MIT licensed.

Paper: https://arxiv.org/abs/2601.23228
Code: https://github.com/ltjed/multiagent-coaching
Blog: https://ltjed.github.io/MAPPA/

If you're building agent systems, would be curious to hear what challenges you're running into with multi-agent coordination.
```

**Angles:** Practical framework, business value (run locally after training), results, invites engagement

---

## 7. LessWrong

**Status:** ✅ Drafted
**Format:** Linkpost with commentary

**Title:**
```
MAPPA: Using AI coaches to train multi-agent systems end-to-end
```

**Body:**

```markdown
**Paper:** https://arxiv.org/abs/2601.23228
**Code:** https://github.com/ltjed/multiagent-coaching
**Blog:** https://ltjed.github.io/MAPPA/

---

Author here. Figured this might be relevant to ongoing discussions about scalable oversight.

## What we did

We have an LLM (Gemini) act as a coach that watches multi-agent systems during training. It scores each action as it happens—process supervision rather than just outcome rewards at the end. The coach sees what each agent produced plus tool outputs (stdout, stderr, errors).

## The parts I find interesting

**You don't need the coach to be able to do the task.** It just needs to tell good actions from bad ones. We found teams trained this way can beat what the coach alone could do on some tasks. Evaluating is easier than solving.

**Credit assignment just... works?** When agent 3 crashes because agent 1 forgot to save a file, the coach checks the filesystem and blames agent 1. No counterfactual rollouts needed. Just look at what actually happened.

**Coach biases leak through.** This one's a bit concerning. Our coach rated regression tasks higher than classification tasks—we didn't program this, it just did. Agents figured this out and started favoring regression. They optimized for the coach, not the task. Expected in hindsight, but worth flagging.

## Numbers

Math (AIME): +5 to +17.5pp
Data science: +16.7pp success, +38% F1, -41% RMSE

## Things I'm still thinking about

What happens when agents get good at gaming the coach? Our coach is stateless—it can't see its own scoring patterns across episodes. A smarter coach might catch this, or might just create more sophisticated failure modes.

Also unclear: how weak can the coach be before this breaks down?

Happy to discuss.
```

**Angles:** Scalable oversight, process supervision, AI training AI, coach limitations, alignment implications

---

## 8. r/reinforcementlearning

**Status:** ✅ Drafted
**Flair:** R (Research)
**Image:** execution_loop_hand-drawn.jpeg

**Title:**
```
[R] Dense process rewards from LLM feedback for multi-agent credit assignment
```

**Body (Reddit Markdown):**

```markdown
We've been working on training multi-agent LLM systems end-to-end with RL. Two problems kept biting us:

**Credit assignment.** Pipeline fails, all agents share the same outcome reward. Agent 3 crashes because Agent 1 forgot to save a file? Both get penalized equally.

**Sparse rewards.** Multi-agent rollouts are expensive—dozens of LLM generations, tool executions, minutes per episode. One scalar at the end is a lot of supervision to leave on the table.

---

## Approach

We use an external LLM as a "coach" that scores each agent action as it happens. The coach sees:
- Agent role and instructions
- Input context
- Agent's output
- Tool feedback (stdout, stderr, errors)

This gives dense per-action rewards without ground truth labels. When something breaks, the coach traces through tool outputs to assign blame correctly.

Train with REINFORCE++ (clipped advantages, no critic needed). Each action gets its own reward signal.

---

## Results

**Math** (3 agents: solver → coder → verifier):
- AIME: +5 to +17.5pp
- AMC: +7.8 to +17.2pp

**Data Science** (3 agents: data engineer → modeler → analyst):
- Success rate: +16.7pp
- Accuracy: +23%
- F1 (classification): +38%
- RMSE (regression): -41%

---

## Links
- **Paper:** https://arxiv.org/abs/2601.23228
- **Code:** https://github.com/ltjed/multiagent-coaching
- **Blog:** https://ltjed.github.io/MAPPA/
- **Twitter:** https://x.com/t_ed_li/status/2019114121250370021

Curious what others think about using LLM judgments as reward signals. The coach is obviously not perfect, but it beats outcome-only rewards for multi-agent setups.
```

**Angles:** RL-focused, credit assignment, dense vs sparse rewards, REINFORCE++, process rewards

---

## 9. Newsletter Pitch

**Status:** ⏳ TODO

---

## 10. WeChat (Chinese)

**Status:** ⏳ TODO

---

## 11. Xiaohongshu (Chinese)

**Status:** ⏳ TODO

---

# Platform Angles

| Platform | Angle |
|----------|-------|
| r/LocalLLaMA | Commercial coach → local team → run offline |
| r/MachineLearning | Novel per-action process rewards for multi-agent RL |
| Hacker News | General fine-tuning pipeline, credit assignment solved |
| LinkedIn | Practical framework for training agent teams |
| LessWrong | Scalable oversight, AI coaching AI |
| Chinese platforms | Works with Qwen/DeepSeek, general pipeline |
