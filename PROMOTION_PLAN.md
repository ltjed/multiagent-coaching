# MAPPA Promotion Plan

## Quick Reference Links

| Resource | URL |
|----------|-----|
| arXiv | https://arxiv.org/abs/2601.23228 |
| HuggingFace | https://huggingface.co/papers/2601.23228 |
| GitHub | https://github.com/ltjed/multiagent-coaching |
| Blog | https://ltjed.github.io/MAPPA/ |
| Tweet | https://x.com/t_ed_li/status/2019114121250370021 |

---

## Key Messaging

### 2-3 Sentence Summary
MAPPA solves two fundamental challenges in end-to-end multi-agent LLM training: **credit assignment** (which agent caused a failure?) and **sample efficiency** (one signal per expensive rollout). An AI coach (Gemini) evaluates every agent action in real-time with process rewards (0-10), examining tool outputs to assign accurate blame without counterfactual reasoning. Demonstrated on math (AIME) and data science (Kaggle-style) benchmarks with 3-agent pipelines.

### Key Angles by Audience

| Audience | Angle |
|----------|-------|
| ML Researchers | Novel process reward mechanism for multi-agent RL; dense feedback without counterfactuals |
| Practitioners | Practical framework for training multi-agent LLM systems end-to-end |
| AI Safety/Alignment | Interpretable credit assignment; AI coaching AI with explicit reasoning |
| Local LLM Enthusiasts | Train smaller models with feedback from larger coaches; open-source code |

---

# Platform Style Guides

---

## Track 1: Twitter/X

### Style & Tone
- **Conversational-professional hybrid** - less formal than LinkedIn/journals
- Use active voice: "We found" not "It was discovered"
- Show genuine enthusiasm without being promotional
- Be authentic - inject personality while maintaining credibility

### Thread Structure (5-7 tweets optimal)

**Tweet 1 - THE HOOK (Critical)**
- Bold statement, surprising finding, or curiosity-inducing claim
- Include thread indicator: 🧵 or "1/" or ⬇️
- Example: "We just released MAPPA. Here's why training multi-agent systems just got way easier:"

**Tweet 2 - The Problem**
- What problem does this solve?
- Keep accessible to non-experts

**Tweets 3-5 - Key Findings/Method**
- One idea per tweet
- Each tweet should be standalone and shareable
- Include visuals (figures, GIFs) - images get **2x algorithmic boost**

**Tweet 6 - Implications**
- What does this mean for the field?

**Final Tweet - Call to Action**
- Links to paper/code
- Tag co-authors, institutions
- End with a question (increases replies by **334%**)

### Visual Strategy
- Key figures from paper (NOT abstract screenshots - hard to read on mobile)
- Architecture diagrams
- Results comparison charts
- GIFs showing algorithm behavior

### Emoji Usage
- Use sparingly: 2-4 per thread maximum
- Recommended: 🧵 (thread), 📄 (paper), ⬇️ (continue), ✨ (highlight)
- Avoid excessive emoji use

### Timing
- **Tuesday-Thursday, 9-11 AM** target timezone
- First-hour engagement determines reach

### Checklist
- [ ] Hook in first tweet with specific/surprising claim
- [ ] Thread indicator (🧵)
- [ ] One idea per tweet
- [ ] At least one compelling visual
- [ ] Co-authors tagged
- [ ] Paper + code links
- [ ] Ends with question
- [ ] 1-2 hashtags (#MachineLearning, #AI)

---

## Track 2: Reddit

### General Reddit Rules
- **90/10 Rule**: 90% genuine contribution, 10% self-promotion
- **Always disclose authorship** upfront - hiding it gets called out
- **Engage with comments** - this dramatically increases reception
- Authors responding turns posts into "mini AMAs"

---

### r/MachineLearning (2.9M+ members)

#### Title Format
```
[R] MAPPA: Training Multi-Agent LLM Systems with Per-Action Process Rewards from AI Feedback
```
- **Required**: `[R]` tag for Research
- Technical, emphasize novelty

#### Post Structure
```
**TL;DR**: 2-3 sentences summarizing core contribution and results

**Problem**: What gap does this address?

**Our Approach**: Brief method description (1 paragraph)

**Key Results**:
- AIME: +5.0-17.5pp improvement
- AMC: +7.8-17.2pp improvement
- Data science: +12.5pp success rate, +30% quality metrics

**Links**:
- Paper: [arXiv]
- Code: [GitHub]
- Blog: [link]

We're the authors - happy to answer questions!
```

#### Tone
- Formal, technical
- Assume ML familiarity
- Emphasize novelty and methodology

#### What Gets Upvoted
- Cutting-edge research with novel approaches
- Open-source code + reproducible results
- Authors who actively engage
- Clear explanations of complex concepts

#### What to Avoid
- Vague hype or sensational claims
- Self-promotion without substance
- Not engaging with comments

---

### r/LocalLLaMA (615K+ members)

#### Title Format
```
MAPPA: Use commercial LLMs to train a TEAM of local agents - then run fully offline
```
- No formal tags required
- Emphasize: commercial coach → local team → run offline

#### Tone
- Practical, conversational, human
- Honest about limitations
- "Your weights, your hardware"

#### Key Angles
- General pipeline for ANY multi-agent system, ANY task, with or without ground truth
- Commercial coach → local team → run offline
- No labeled data needed - coach provides training signal
- Works with Qwen, LLaMA, DeepSeek

#### Post (Final Draft)
See `CONTENT_DRAFTS.md` for full post.

Summary:
- Lead with "commercial coach to train local team, run offline"
- Explain credit assignment problem casually
- Give both results (data science +16.7pp/+38% F1, math +17.5pp)
- Emphasize general framework
- End with "Ask me anything"

---

### r/reinforcementlearning (~100K members)

#### Tone
- **High technical depth** expected
- Assume familiarity with PPO, policy gradients, reward shaping
- Mathematical notation acceptable
- Reference relevant prior work

#### Post Structure
```
[Paper] MAPPA: Per-Action Process Rewards for Multi-Agent RL

We address credit assignment and sample efficiency in multi-agent LLM training using REINFORCE++ with dense process rewards from an external LLM coach.

**Key insight**: Instead of outcome-based rewards, an LLM coach (Gemini) evaluates each agent action by examining:
- Agent's observation space
- Generated output
- Tool feedback (stdout, stderr, errors)

This enables accurate credit assignment without counterfactual reasoning.

**Algorithm**: REINFORCE++ with global batch advantage normalization, DeepSpeed + Ray distributed training

**Results**:
- AIME: +5.0-17.5pp
- AMC: +7.8-17.2pp
- DSBench: +12.5pp success rate

Paper: [arXiv] | Code: [GitHub]

Author here - happy to discuss the RL details!
```

---

## Track 3: Hacker News

### Title Format
```
Show HN: MAPPA – Train multi-agent LLM systems with AI coaching
```
- Matter-of-fact and descriptive
- Plain language, no marketing speak
- Under 80 characters

### Tone
- **Technical and direct** - share implementation details
- **Humble but confident** - acknowledge limitations, stand behind work
- **Data-driven** - back claims with evidence
- **Authentic** - personal > corporate (2.4x advantage)

### What Works
- "I built this because X frustrated me. Here's what I learned..."
- Immediately try-able demos
- Open source with good documentation
- Unique application (not "another chatbot")

### What Gets Flagged/Downvoted
- Marketing language or press release tone
- Signup pages without working product
- Hidden self-promotion
- Vote manipulation (even 5-6 friends upvoting triggers detection)
- Defensive responses to criticism

### Optimal Timing
- **Best**: Sunday 6 AM UTC (Saturday 11 PM PDT)
- Weekend posts have 20-30% better breakout rate
- Posts need 4-6 upvotes within 30 minutes to reach front page

### First Comment (Critical)
Post immediately after submitting:
```
Author here. Built this because credit assignment in multi-agent RL is painful - when a 3-agent pipeline fails, which agent broke it?

Our approach: have an LLM coach (Gemini) watch each agent's actions and tool outputs, assigning blame in real-time. Sounds simple but enables dense feedback without counterfactual reasoning.

Limitations:
- Requires decent GPU setup (2-8x 80GB)
- Coach API calls add latency
- Currently tested on Gemini; other coaches should work but untested

Happy to discuss the technical details!
```

### Engagement Rules
- Be present for first 3-4 hours
- Comments are stronger ranking signal than upvotes
- Never be snarky or condescending
- Address arguments, not people
- Say "good point, I hadn't considered that" when appropriate

### Pre-Launch Checklist
- [ ] CDN configured (sites crash without this)
- [ ] Working demo without signup
- [ ] GitHub repo with good README
- [ ] First comment drafted
- [ ] Available to respond for 4 hours

---

## Track 4: LessWrong / AI Alignment Forum

### Community Norms
- **Truth-seeking over persuasion** - focus on evidence, not rhetoric
- **Epistemic transparency** - be explicit about uncertainty
- **Show reasoning chain** - not just conclusions

### Required: Epistemic Status Header
Start every post with:
```
Epistemic status: [Your confidence level and caveats]
```
Examples:
- "Epistemic status: Fairly confident about the technical claims, more speculative about alignment implications"
- "Epistemic status: Preliminary results, would welcome pushback"

### Post Structure
```
# MAPPA: Per-Action Process Rewards for Multi-Agent Systems

**Epistemic status**: Confident about technical results; alignment implications are more speculative.

## Summary
[2-3 sentence TL;DR]

## Alignment Relevance
[Why this matters for alignment - THIS IS CRUCIAL]
- Credit assignment is a core alignment problem
- External coach provides interpretable feedback
- AI evaluating AI with explicit reasoning traces

## Technical Overview
[Accessible explanation of method]

## Key Results
[Numbers and benchmarks]

## Limitations
[Be honest - community values this]

## Open Questions
[What remains unsolved]

## Links
Paper: [arXiv] | Code: [GitHub]
```

### Framing for Alignment Audience
Address explicitly:
- How does this help with **outer alignment** (specifying what we want)?
- How does this help with **inner alignment** (model actually pursuing that)?
- What are the **capabilities vs. safety tradeoffs**?
- Does this advance capabilities more than safety? (Central concern)

### Alignment Angles for MAPPA
1. **Interpretable credit assignment** - coach provides explicit reasoning for blame
2. **AI oversight of AI** - larger model supervising smaller models' actions
3. **Dense feedback without human labels** - scalable oversight pattern
4. **Audit trail** - file-based coordination creates inspectable artifacts

### Topics That Resonate
- Mechanistic interpretability
- Deception detection
- Scalable oversight
- Verification and evaluation

### What to Avoid
- Capabilities advances without safety angle
- LLM-generated content without heavy editing
- Overly promotional tone

---

## Track 5: LinkedIn

### Tone
- **Professional but accessible** - not boring
- Show human side - authentic voice
- Express genuine excitement without hype
- Credibility + approachability

### Post Structure
```
[Emoji] Exciting news! Our paper "MAPPA: Multi-Agent Systems with Per-Action Process Rewards" is now available.

[1-2 sentences: What it does in accessible terms]

[1-2 sentences: Why it matters / practical applications]

Key contributions:
• [Point 1]
• [Point 2]
• [Point 3]

Grateful to my co-authors [tag them] and [institution].

[Question to encourage comments]

#MachineLearning #AI #MultiAgentSystems #ReinforcementLearning

Link in comments 👇
```

### Format Guidelines
- **Length**: 200-400 words (1,300-2,000 characters)
- **First 140 characters critical** - this is the "See more" cutoff
- Use line breaks liberally
- Bold for key points
- Bullet points for findings

### Engagement Strategy
- **PUT LINKS IN COMMENTS** - external links reduce reach significantly
- Tag co-authors and institutions
- Use 3-5 hashtags maximum
- End with a question
- Respond to comments quickly (first hour matters most)

### Emoji Usage
- 2-4 maximum
- Recommended: 📢 (announcement), 🔬 (research), 💡 (insight), ✅ (findings), 👇 (link below)

### Timing
- Tuesday-Thursday, working hours for short posts
- Evenings for long-form content

---

## Track 6: Chinese Platforms

---

### WeChat Moments (朋友圈)

#### Platform Context
- **67% of Chinese researchers** use WeChat to discover research
- **56%** share their work on the platform
- Semi-closed networking based on acquaintances

#### Content Format
- Convert conclusions into **popular science posts**
- Accompany text with **flowcharts or data visualizations**
- Focus on **novelty** - new findings are popular
- Provide **authentic scientific information with evidence**

#### Visual Presentation
- Add filters and design thoughtful layouts
- Use infographics and flowcharts
- Posts should appear well-designed

#### Distribution Strategy
- Share through personal Moments AND WeChat Official Accounts
- Leverage group chats for academic communities
- Create QR code posters linking to articles
- Invite reputable users to reshare

#### Sample Post (Chinese)
```
📢 新论文发布！

我们提出了MAPPA：一种用于多智能体系统的端到端训练方法。

核心创新：
• AI教练（Gemini）实时评估每个智能体的动作
• 解决多智能体系统中的信用分配问题
• 无需人工标注即可获得密集反馈

结果：
• AIME数学竞赛：+17.5pp
• 数据科学任务：+12.5pp成功率

论文链接：[arXiv]
代码开源：[GitHub]

欢迎讨论交流！
```

---

### Xiaohongshu / 小红书

#### Platform Context
- **300M+ monthly active users**
- Younger demographics
- Users actively **search for answers**
- Visual-first platform

#### Image Specifications
| Type | Size | Ratio |
|------|------|-------|
| Vertical Cover | **1242 x 1660 px** | 3:4 (recommended) |
| Square | 1080 x 1080 px | 1:1 |
| Video | 1080p minimum | 9:16 |

#### Cover Design Rules
- **Text must occupy 70%+** of visual space
- Use **3-4 different font sizes** for hierarchy
- Main title **3x larger** than subtitles
- Extract **2-3 keywords** for special treatment
- Place title in center/prominent position

#### Visual Style for Tech Content
- Grid paper backgrounds for programming/logic topics
- Colors: deep blue, orange, bright yellow, green
- Maintain consistent branding across posts
- Embrace authenticity over heavy polish

#### Content Strategy
- **Photo carousels: 4-8 images**
- Detailed captions for each image
- **5-10 topic hashtags** at end
- ~700 characters text
- Conversational tone with platform-style emoji

#### Algorithm Notes
- **First 1-4 hours critical** - early engagement determines viral potential
- **Saves** are highest-value engagement (long-term preservation)
- Include 3-5 relevant keywords in captions

#### Hashtags
```
#AI #人工智能 #机器学习 #多智能体 #强化学习 #LLM #开源项目 #科研分享
```

---

### Zhihu (知乎)

#### When to Use
- Deep technical discussions
- Targeting industry professionals + academics
- AI ethics, labor impact, societal implications
- Less optimal for quick announcements (use WeChat)

#### Style
- Long-form, in-depth technical explanations
- Include practical code examples
- Reference specific algorithms and comparisons

---

## Track 7: Newsletters & Aggregators

### Information Package

```
Paper Title: MAPPA: Multi-Agent Systems with Per-Action Process Rewards from AI Feedback

Authors: Ed Li, Junyu Ren, Cat Yan (Yale University)

Links:
- arXiv: https://arxiv.org/abs/2601.23228
- GitHub: https://github.com/ltjed/multiagent-coaching
- Blog: https://ltjed.github.io/MAPPA/

Summary:
MAPPA introduces per-action process rewards from an AI coach to solve
credit assignment and sample efficiency in multi-agent LLM training.
Unlike outcome-based rewards, the coach evaluates every agent action
by examining tool outputs and artifacts, enabling accurate blame
assignment without counterfactual reasoning.

Results: +17.5pp on AIME, +12.5pp on data science tasks.
```

### Target Newsletters

| Newsletter | Focus | Submission Method | Status |
|------------|-------|-------------------|--------|
| The Batch (Andrew Ng) | General ML | research@deeplearning.ai | [ ] |
| Import AI (Jack Clark) | AI research & policy | Submission form | [ ] |
| The Gradient | ML research | thegradiented@gmail.com | [ ] |
| Papers with Code | ML papers | Auto-indexed, can submit | [ ] |
| AI Weekly | General AI | submission form | [ ] |
| Last Week in AI | AI news roundup | newsletter@lastweekin.ai | [ ] |
| Davis Summarizes Papers | Paper summaries | Twitter DM | [ ] |
| MLOps Community | Practical ML | Slack channel | [ ] |

---

## Track 8: Academic Platforms

### HuggingFace
- [x] Paper already on HuggingFace Papers
- [ ] Engage with comments/discussions
- [ ] Consider adding model weights

### Papers With Code
- [ ] Ensure paper is indexed
- [ ] Link GitHub repository
- [ ] Add benchmark results

### Semantic Scholar
- [ ] Verify paper is indexed
- [ ] Check author profiles linked

### Google Scholar
- [ ] Verify indexing
- [ ] Add to author profiles

---

## Execution Timeline

### Week 1: Foundation
| Day | Tasks |
|-----|-------|
| Day 1 | Post Twitter thread, record URL |
| Day 1 | Post on LinkedIn (link in comments) |
| Day 2 | Submit to r/MachineLearning |
| Day 2 | Submit to Hacker News (Sunday 6AM UTC optimal) |
| Day 3 | Submit to r/LocalLLaMA |
| Day 3 | Begin newsletter outreach |

### Week 2: Expansion
| Day | Tasks |
|-----|-------|
| Day 8 | Post to r/reinforcementlearning |
| Day 9 | Post to r/singularity |
| Day 10 | Submit to LessWrong |
| Day 11 | WeChat and Xiaohongshu posts |
| Day 12-14 | Twitter amplification outreach |

### Ongoing
- Monitor all platforms for questions/comments
- Engage with discussions (first hours critical)
- Track metrics

---

## Content Drafts Checklist

### High Priority
- [ ] Twitter thread (English)
- [ ] r/MachineLearning post
- [ ] r/LocalLLaMA post
- [ ] Hacker News submission + first comment
- [ ] Newsletter pitch template

### Medium Priority
- [ ] LinkedIn post
- [ ] r/reinforcementlearning post
- [ ] r/singularity post
- [ ] LessWrong post

### Lower Priority
- [ ] WeChat moments post (Chinese)
- [ ] Xiaohongshu carousel + post (Chinese)

---

## Visual Assets Needed

- [ ] Logo (exists: assets/logo.png)
- [ ] Architecture diagram (training loop visualization)
- [ ] Results comparison chart
- [ ] Social media card (1200x630 for Twitter/LinkedIn)
- [ ] Xiaohongshu carousel images (1242x1660, 4-8 images)

---

## Metrics to Track

| Platform | Metrics |
|----------|---------|
| GitHub | Stars, forks, issues, traffic |
| arXiv | Downloads, citations |
| HuggingFace | Paper likes, comments |
| Twitter | Impressions, retweets, replies |
| Reddit | Upvotes, comments, cross-posts |
| Hacker News | Points, comments, front page position |
| Blog | Page views |

---

## Talking Points for Q&A

1. **"How is this different from RLHF?"**
   - RLHF uses outcome rewards; MAPPA uses per-action process rewards
   - Coach examines tool outputs for each action, not just final result

2. **"Why not just use a critic network?"**
   - Critic networks struggle with credit assignment across agents
   - External coach can examine artifacts and tool outputs directly

3. **"What's the compute cost?"**
   - 2-8x 80GB GPUs for training
   - Coach API calls add latency but provide dense signal
   - More efficient than sparse outcome rewards

4. **"Can this work with other coaches?"**
   - Yes, designed for any LLM coach
   - Tested with Gemini; others should work

5. **"Does this advance capabilities or safety more?"** (For alignment community)
   - Interpretable credit assignment aids oversight
   - Coach reasoning is explicit and auditable
   - File-based coordination creates inspection trail

---

## Notes & Decisions Log

| Date | Decision/Note |
|------|---------------|
| | |

---

# Appendix: Sample Data from Platform Research

## Samples Collected Summary

| Platform | Samples | Data Quality |
|----------|---------|--------------|
| r/LocalLLaMA | 20+ posts with upvotes | High - year in review data |
| r/MachineLearning | ~5 patterns identified | Medium - general patterns |
| Hacker News | 10+ Show HN posts | Medium - titles and descriptions |
| LessWrong | 10+ posts | Medium - topics and structures |
| Chinese Platforms | Comprehensive | High - detailed style guides |
| Twitter/X | Limited | Low - likes now private |
| LinkedIn | Limited | Low - hard to search |

---

## Current Trends (Jan-Feb 2026)

### What's Viral RIGHT NOW

**Claude Code Dominance**
- Claude Code went viral during winter holidays 2025-2026
- "Vibe coding" trend - non-programmers building apps
- Meme: "I'm not joking and this isn't funny" (Google engineer saying Claude Code rebuilt their year-long project in an hour)
- "Ralph Wiggum" technique (bash loop for self-healing code) went viral
- Cowork was built by Claude Code itself in 1.5 weeks - "the memes write themselves"

**DeepSeek V4 Hype**
- r/LocalLLaMA and r/Singularity in "Code Red" mode
- Expected mid-February 2026
- Reportedly outperforms Claude and GPT-4/5 in coding
- January 2025 DeepSeek R1 memes still referenced (NVIDIA $600B crash)

**Qwen Taking Over LocalLLaMA**
- "Qwen has been taking over the LocalLlama subreddit"
- "For the time being, the default is Qwen"
- Qwen 3 leads in math (92.3% on AIME25) and coding

### Current Meme Formats (2026)

| Meme | Context | Usage |
|------|---------|-------|
| "I'm not joking and this isn't funny" | Google engineer on Claude Code | Serious capability announcements |
| "Ralph Wiggum with Claude Code" | Bash loop technique | Agentic coding discussions |
| "Claudestrophobic" | Fear of AI progress | AI anxiety commentary |
| "Associate Claude Operator" | New job titles | AI workforce disruption |
| DeepSeek "identity crisis" | Introduces as GPT-4, then corrects | Model comparison jokes |
| "$6 million vs $1 trillion" | DeepSeek training cost | Efficiency vs big tech |

### What's HOT on r/LocalLLaMA Now

1. **DeepSeek V4 tracking** - "Can I run this on my rig?"
2. **Qwen 3 discussions** - Default recommendation
3. **Claude Code vs Cursor debates** - "Cursor's Dead and Claude Code Killed It"
4. **Quantization talk** - FP8/INT4 for running big models locally
5. **Chinese model revolution** - DeepSeek, Qwen, Kimi K2

---

## r/LocalLLaMA Historical Data (2024)

### Top Posts (For Pattern Reference)

| Upvotes | Title/Topic | Type |
|---------|-------------|------|
| 3,399 | "If I Can't Run It on 3090" meme | Meme |
| 1,864 | 14x RTX 3090 build | Hardware |
| 1,586 | Roasting OpenAI | Commentary |
| 1,481 | 4x RTX 4090 build | Hardware |
| 1,341 | LLaMA 3 training news | News |
| 1,281 | LLaMA 3.3 70B release | Model Release |
| 1,208 | Bitnet architecture | Technical |
| 1,181 | Gemma first release | Model Release |

### Enduring Patterns

**What consistently performs:**
1. **Memes** - Still highest engagement
2. **Hardware builds** - GPU flex posts
3. **Major model releases** - Open-weight models
4. **Anti-corporate commentary** - Now extended to "Chinese models beating Big Tech"
5. **"Can I run it locally?"** - Perennial question

**Current tone that works:**
- Irreverent, anti-corporate (especially anti-OpenAI)
- "GPU-poor friendly" solidarity
- Excitement about Chinese open-source models
- "DeepSeek trained for $6M" energy
- Honest about hardware requirements

---

## Hacker News Show HN Samples (ML/AI)

### Recent Show HN Posts

| Title | Topic | Key Success Factor |
|-------|-------|-------------------|
| "Show HN: 32KB deductive engine that catches LLM hallucinations" | LLM tooling | Small footprint, practical utility |
| "Show HN: General Intelligence – Active knowledge framework for ML" | Framework | Novel approach, pip-installable |
| "Show HN: ML model to predict 66.45% of NBA games" | Applied ML | Fun application, specific metric |
| "Show HN: Deep-ML Labs: Learn ML by building from scratch" | Education | Hands-on, practical |
| "Show HN: JavelinGuard: Low-Cost Transformer Architectures" | Security | Specific use case, benchmarked |
| "Show HN: Visualizing 8k+ LLM papers with t-SNE" | Visualization | Unique insight, visual |
| "Show HN: LLM Skirmish – benchmark where LLMs play RTS games" | Benchmark | Novel, engaging format |
| "Show HN: Noether – ML framework for physical engineering" | Framework | Niche domain, clear value |

### Pattern Analysis

**What works on HN:**
1. **Specific, measurable claims** - "66.45% accuracy", "32KB", "8k+ papers"
2. **Immediately try-able** - pip install, web demo, CLI tool
3. **Novel applications** - not "another chatbot"
4. **Open source** - always mention
5. **Technical depth available** - ready to discuss implementation

**Title patterns:**
- State what it does in plain language
- Include a concrete metric or scale
- No marketing language or hype

**First comment essentials:**
- Backstory: "Built this because X frustrated me"
- What's different about the approach
- Honest limitations upfront
- Ready to discuss technical details

---

## LessWrong/AI Alignment Forum Samples

### Notable Posts (2024)

| Title | Topic | Engagement Pattern |
|-------|-------|-------------------|
| "Interpretability's Alignment-Solving Potential: Analysis of 7 Scenarios" | Interpretability | Scenario-based analysis |
| "If interpretability research goes well, it may get dangerous" | Dual-use concerns | Contrarian take |
| "Research Areas in Interpretability (UK AISI)" | Funding/research | Institutional backing |
| "Self-Other Overlap: A Neglected Approach to AI Alignment" | Novel technique | Practical proposal |
| "How I'd like alignment to get done (as of 2024-10-18)" | Meta-strategy | Personal perspective |
| "An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers v2" | Curation | Opinionated, practical |

### Pattern Analysis

**What resonates on LessWrong:**
1. **Explicit alignment framing** - always connect to safety
2. **Scenario analysis** - structured reasoning
3. **Opinionated takes** - "extremely opinionated" in title works
4. **Practical proposals** - not just theory
5. **Contrarian perspectives** - "may be dangerous" gets attention

**Structure patterns:**
- Epistemic status header (always)
- Clear section hierarchy
- Numbered scenarios or frameworks
- Limitations section
- "What would change my mind"

**Topics with highest engagement:**
- Mechanistic interpretability
- Scalable oversight
- Deception detection
- Capabilities vs safety tradeoffs

---

## Chinese Platforms Samples (Xiaohongshu)

### Platform Statistics
- **16.15 million** posts tagged "AI" (more than "makeup" at 10.06M)
- **17x growth** in AMA-related content over 3 months
- **3.7 billion views** on ask-me-anything content
- **50,000+ developers** registered

### High-Engagement Content Types

| Type | Engagement | Example |
|------|------------|---------|
| AMA format | Very High | Researchers answering questions live |
| Paper explanations | High | ICLR/ACL/CVPR paper breakdowns |
| Career advice | High | PhD daily life, job hunting |
| Tool tutorials | Medium-High | How to use specific AI tools |
| Hardware builds | Medium | GPU setups, local deployment |

### Visual Style That Works

**Cover Design:**
- 3:4 aspect ratio (1242×1660px)
- Text occupies 70%+ of visual space
- 3-4 font sizes for hierarchy
- High saturation colors
- "Black background + fluorescent text" for tech

**Carousel Format:**
- 4-9 images optimal
- 7-9 images = 37% more saves
- Detailed captions per image
- Consistent branding across slides

### Successful Post Example (Structure)

```
📢 [Hook - What the paper solves]

🔬 [Paper name + conference]

核心创新 (Core Innovation):
• Point 1
• Point 2
• Point 3

结果 (Results):
• Metric 1
• Metric 2

💡 [Why it matters to you]

#AI #机器学习 #论文解读 #深度学习 #科研分享
```

### Notable Participants in AI AMAs
- Kai-Fu Lee (01.AI)
- Zhiyuan Liu (Tsinghua/ModelBeast)
- Qi Yin (Megvii founder)
- Researchers from: Alibaba Qwen, Tencent Hunyuan, Moonshot AI
- Thomas Wolf (HuggingFace co-founder)

---

## Twitter/X Limitations

**Data Collection Challenges:**
- Likes made private (June 2024)
- Engagement metrics not publicly searchable
- Algorithm changes frequently

**Known Successful Patterns (from guides):**
- Threads: 5-7 tweets optimal
- Images: 2x algorithmic boost
- Questions at end: +334% replies
- "Retweet" asks: +311% RTs
- Tuesday-Thursday 9-11 AM: best timing

**Notable ML Accounts to Study:**
- @kaborge (Andrej Karpathy)
- @ylecun (Yann LeCun)
- @AndrewYNg (Andrew Ng)
- @sama (Sam Altman)
- @anthropicai
- @OpenAI
- @GoogleDeepMind

---

## Key Insights from Sample Data

### Cross-Platform Patterns

| Factor | High Engagement | Low Engagement |
|--------|-----------------|----------------|
| **Tone** | Authentic, opinionated, honest | Marketing-speak, hype |
| **Claims** | Specific metrics, honest limitations | Vague superlatives |
| **Format** | Visual, structured, scannable | Wall of text |
| **Timing** | First few hours critical | Random posting |
| **Engagement** | Author responds to comments | Post and ghost |

### Platform-Specific Success Formulas

**r/LocalLLaMA**: Meme + anti-corporate + practical = viral

**Hacker News**: Specific claim + try it now + humble tone = front page

**LessWrong**: Alignment framing + epistemic status + scenarios = karma

**Xiaohongshu**: Visual carousel + AMA format + emoji + hashtags = saves

### Gaps in Data

- Twitter: Need to manually analyze recent ML paper threads
- LinkedIn: Hard to search for specific engagement data
- r/MachineLearning: Need to scrape actual [R] posts with upvotes

---

## Recommendations Based on Samples

### For MAPPA specifically:

**r/LocalLLaMA angle (UPDATED for 2026)**:
- Lead with "train smaller models using larger coach feedback" (fits Qwen/DeepSeek ecosystem)
- Reference current meta: "Works with Qwen, DeepSeek, any open model"
- Acknowledge GPU requirements honestly ("2-8x 80GB - yes, beefy - but open source")
- Position as "train your own multi-agent system without OpenAI/Anthropic API costs"
- Potential meme angle: "When your AI coach catches the bug your 3-agent pipeline missed"

**Hacker News angle**:
- Title: "Show HN: MAPPA – Train multi-agent LLM systems with per-action AI coaching"
- First comment: credit assignment pain point, what's different, limitations
- Mention "open source, MIT license" prominently
- Reference current interest in agentic systems (Claude Code, OpenClaw)

**LessWrong angle**:
- Frame as scalable oversight mechanism
- Connect to current "AI coaching AI" discourse (Claude Code built Cowork)
- Discuss interpretability of coach reasoning traces
- Address: "Does training multi-agent systems with AI feedback accelerate capabilities or safety more?"

**Xiaohongshu angle**:
- Create visual explainer carousel (4-8 images)
- Consider AMA format - very hot right now with 17x growth
- Connect to current Chinese AI momentum (Qwen, DeepSeek)
- Use bilingual keywords for discoverability

**Twitter/X angle (UPDATED)**:
- Hook into Claude Code discourse: "What if you could train your agents like Claude Code trains itself?"
- Reference DeepSeek efficiency narrative: "Dense feedback without $1T training budgets"
- Visual: architecture diagram showing coach evaluating agent actions

### Current Cultural Hooks to Consider

| Hook | Platform | Why It Works Now |
|------|----------|------------------|
| "AI coaching AI" | All | Claude Code built Cowork; resonates |
| "Open-source training" | LocalLLaMA, HN | Anti-OpenAI, pro-efficiency sentiment |
| "$6M not $1T" | Twitter, Reddit | DeepSeek efficiency narrative |
| "Credit assignment solved" | ML communities | Technical pain point |
| "Scalable oversight" | LessWrong | Alignment framing |
| Chinese model compatibility | Xiaohongshu | Qwen/DeepSeek momentum |
