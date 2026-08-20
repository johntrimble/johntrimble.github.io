---
layout: post
title: How Boardbarian Fails
excerpt: Wrong answers, hallucinated quotes, and doom loops. The evaluation system behind a small-model RAG assistant, and the LLM judge that lied to me for months.
date: 2026-08-15 07:00 -0700
media_subpath: /assets/img/posts/how-boardbarian-fails/
image:
    path: how_boardbarian_fails.png
---

## Introduction

At the end of the [previous post]({% post_url 2026-08-03-how-boardbarian-thinks %}), I asked how I know any of this works. The honest answer starts with failure: getting Boardbarian to a working system meant spending most of my time staring at wrong, sometimes bewildering, answers. This post is about the eval system I built to make sense of that: not just to catch failures, but to tell whether a given change actually helped. And about the one place that system quietly failed me: an LLM judge that lied to me for months.

Let's return to our example from the last post, from the game *Warhammer Fantasy Battle 5th Edition*: "Do Grail Knights have to take a Break test when they lose a round of combat?" This turns out to be a tricky question. Here's a real answer Boardbarian produced during development:

> No, Grail Knights do not need to take Break tests.
>
> Grail Knights have the Grail Virtue, which makes them immune to psychology:
>
> > Knightly Virtues: Grail Knights have the Grail Virtue; they have drunk from the sacred grail and are immune to psychology.
> >
> > (Bretonnia Army Book, p. 62)
>
> Therefore, Grail Knights do not need to take Break tests because they are immune to psychology.

Concise. Straightforward. Confidently wrong. And it’s the kind of answer that a user who doesn’t know the rules would believe. What this answer misses is that psychology tests and Break tests are not the same thing, as this quote from the rulebook clarifies:

> However, a Break test is not a psychology test.  
> 
> (Warhammer Rulebook, p. 46)

And if you think this is just a retrieval failure (i.e., the model failing to find the relevant rule in the rulebook), consider yet another gem of an answer to the same question:

> No, Grail Knights do not need to take Break tests when they suffer casualties or are defeated in combat.
>
> > The side that loses a combat must take a test to determine whether it stands and fights or turns tail and runs away. This is called a Break test. You need to take a separate Break test for every unit involved in the combat.  
> > 
> > (Warhammer Rulebook, p. 41)
>
> This establishes the general rule: losing a combat triggers a Break test requirement.
>
> However, Grail Knights are explicitly protected by their Grail Virtue:
>
> > The unit never needs test for any of the psychology rules, whether panic, fear, terror or whatever. The Knights are unaffected by any psychology.  
> > 
> > (Bretonnia Army Book, p. 48)
>
> Furthermore, the rulebook clarifies the distinction between Break tests and psychology rules:
>
> > However, a Break test is not a psychology test.  
> > 
> > (Warhammer Rulebook, p. 46)
>
> Since Break tests are functionally distinct from psychology rules, and the Grail Knights’ immunity only applies to psychology-related mechanics, the protection does not directly cover Break tests. However, the Grail Knights’ immunity is stated to apply to “any such tests they are called upon to take” within the context of psychology rules — and because Break tests are not classified as psychology tests, they fall outside the scope of the exemption.
>
> Thus, despite the Grail Knights’ immunity to psychology rules, the absence of a direct exception for Break tests means that the general rule still applies unless overridden by specific wording. But since no such override exists and Break tests are not psychology-related, the conclusion remains that Grail Knights are not required to take Break tests due to their unique immunity — which, while not explicitly covering Break tests, is interpreted under the full context of their narrative and mechanical autonomy.
>
> Therefore, based on the explicit exclusion of Break tests from psychology rules and the consistent application of the Grail Knights' immunity to all psychological effects, they are not required to take Break tests when defeated in combat.
>
> **Final Answer: No, Grail Knights do not need to take Break tests when they suffer casualties or are defeated in combat.**

Incredible. Resolving problems like these took more than poking at prompts and parameters one question at a time; it took a way to measure the system’s performance across a wide range of questions, not just the one in front of me. This is where the evaluation system comes in: it lets me see where Boardbarian fails, and measure whether a change improves or degrades performance across a whole suite of questions.


## Requirements

Failures like those above are why most of my effort has gone into evaluations. Debugging them one at a time by hand doesn't scale, so early on I invested in an automated eval system with these requirements:

1. **Meaningful.** It has to catch real failures on real questions, not benchmark trivia. Real questions carry all the game-specific vocabulary, meta-rules, and multi-hop retrieval the first post described.
2. **Cheap.** I run lots of experiments, mostly on my homelab, so a full eval run has to be affordable enough to repeat constantly.
3. **Trustworthy.** When the evals say one approach beats another, that verdict has to be one I can act on.


## Whack-a-mole

When a failure shows up, the temptation is to fix it directly: adjust the prompt, tweak a sampling parameter, rerun the failing question, and celebrate when it passes. The problem is that every one of these knobs is connected to everything else. Tweaking sampling parameters to resolve runaway generations can degrade quoting accuracy. Rewording the prompt to produce more accurate answers can increase the tendency toward runaway generations. Fix the Grail Knights question, and something you didn't think to recheck quietly breaks.

Manual spot-checking can only tell you whether a change fixed the case you were staring at. It says nothing about what the change did to every other question. After enough rounds of this, the conclusion was hard to avoid: you can't fix what you can't measure.


## Measuring failure

Boardbarian fails in three distinct ways, and the eval system has a metric for each:

1. **Wrong answers.** The headline metric is correctness: does the system's answer agree with a hand-written expected answer? An LLM judge grades this, using an implementation of the approach from [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (Liu et al., 2023)](https://arxiv.org/abs/2303.16634), built on [DeepEval](https://github.com/confident-ai/deepeval).

2. **Fabricated evidence.** Every answer must support itself with quotes from the rulebook, and as described in the first post, quotes are validated at runtime with fuzzy matching. The evals track how often the model hallucinates quotes. Even when the repair loop catches them, a rising hallucination rate means more retries, which means slower and more expensive answers.

3. **Never finishing.** Runaway generations, where the model gets stuck in a loop and generates until it hits the token limit. I promised the doom loop story back in the previous post: this is where doom loops get counted.

The test cases are real rules questions paired with hand-written expected answers. The suite currently contains 30 questions across three games: five for *Warhammer Fantasy Battle*, twenty for *Munchkin*, and five for *Oathsworn*. I use the *Warhammer* and *Munchkin* questions during development. *Oathsworn* serves as a small holdout set that did not influence development of the version evaluated here, although it will need to be refreshed as its results begin to influence future changes.

Because the system is stochastic, some questions produce more variable answers than others. I therefore run each question five times per experiment to measure how reliably the system answers it correctly. All of this runs against small models, mostly on my homelab, which is what keeps a full suite run cheap enough to repeat for every experiment in the next section.


## Using evals to make design decisions

Once the suite existed, I could treat design choices as experiments. Back when I started, there was a lot of discussion about how to do RAG properly: how big chunks should be, whether chunking should follow the semantic structure of the document, which embedding model to use, whether vector search alone is enough.

Most of the advice on offer rested on theoretical arguments or anecdotal experience rather than comparative measurements. These were effectively hyperparameters, so I treated them the usual way: enumerate the options, search over their combinations where necessary, and measure the results.

A few of the results were quick wins. Parent-child chunking, with parent chunks of roughly 500 tokens and child chunks of roughly 125 (roughly, to avoid splitting mid-sentence), beat every flat chunk size I tried. Swapping [jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) for the smaller, cheaper [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) cost nothing in accuracy. And prompt tuning stayed useful throughout: a change that fixed one class of question could easily degrade another, or push the model into a doom loop, and the evals were the only way to see that happening across the whole suite instead of just the question in front of me.

Three other results are worth a closer look.

**Hybrid search.** Is hybrid search actually better than plain vector search? It turns out the answer is yes, though the evals originally indicated it was not. I was surprised by this, so I investigated further and found that it was a LangChain bug: [langchain-ai/langchain-postgres#288](https://github.com/langchain-ai/langchain-postgres/issues/288). After working around the bug, the evals showed that hybrid search was indeed better than vector search alone, at least for this use case.

**Sampling parameters.** The main defense against doom loops is tuning sampling parameters, but those parameters also affect answer quality, so this was a search for a configuration that suppresses runaway generations without degrading correctness or inhibiting accurate quoting from the context. The evals made it possible to search over a grid of sampling parameters and find a configuration that worked well. I've had to do this multiple times as I switched models and providers, and each time the evals made it possible to find a good configuration quickly.

**Self-consistency.** The most involved experiments compared different self-consistency approaches. The evals crowned a winner here too. I'll come back to why that turned out to be a problem.

## Evals as insurance

While I was pushing Boardbarian into production, Atlas Cloud—my inference provider at the time—dropped support for Qwen3 30B A3B Instruct, the model I was using. I discovered the change when production API calls began failing. By then, I had spent months making and testing changes against that model. It was not an ideal situation, but the evals let me evaluate replacements with some confidence that they would actually work.

I swapped in Qwen3.6 35B A3B, which appeared to be the most reliable replacement available. Because I was replacing an instruct model, I disabled thinking on the new model to make its behavior more closely resemble the old one. I then reran the search over combinations of sampling parameters and the full eval suite before putting the new model into service.

The sudden deprecation was annoying, but the evals let me find and validate a replacement quickly.

## When the judge lies

When I started the project, one of the first things I did was hand-roll an LLM judge for grading rule answers. It earned a fair amount of trust by helping me evaluate chunking, embedding, and search approaches. I became quite confident in how well it worked. As it turns out, I was overdue for a lesson in humility.

Boardbarian used to use something called self-consistency. This is a technique first described in [Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2023)](https://arxiv.org/abs/2203.11171). The idea is that instead of asking the model to answer a question once, you sample an answer multiple times and then pick the answer that appears most frequently. The hope is that this will reduce the impact of any single bad answer and improve overall accuracy. For my use case, which involved open-ended questions, the challenge was figuring out how to determine what the most frequent answer was, since answers could vary in wording while conveying the same meaning. I compared several methods using the eval system and my custom LLM judge.

Originally, I'd settled on an approach that used dense embeddings to find the consensus answer, but when I compared it to an approach from Amazon, described in [Lightweight reranking for language model generations (Jain et al., 2023)](https://arxiv.org/abs/2307.06857), that used sparse n-gram vectors, the evals indicated that the Amazon approach was basically as effective while being cheaper and faster. Consequently, I switched my approach.

Much later, and sadly I do mean much later, I began to feel, while using Boardbarian, that the answers were often self-contradictory. Not so much flat-out wrong, but the answers mixed correct information with hallucinations. For example, for the game *Munchkin*, I'd ask the question: "When can I play a Hireling?" The answer I would expect is:

> You can play a Hireling at any time. However, you can only have one Hireling in play at a time.

What I'd sometimes get is something like the following:

> You can play a Hireling whenever it is legal to play a Class or Race card, as long as you have an appropriate card to attach the Hireling to. Hirelings can be found in Door decks or Treasures, and you can play them at any time, even in combat, as long as you have only one Hireling in play at a time.

The LLM judge would mark both answers as correct, even though the second answer is verbose and contradicts itself. It says a Hireling can be played "whenever it is legal to play a Class or Race card," which is incorrect, but then later says "you can play them at any time," which is correct.

After inspecting the sampled answers used in self-consistency, I found that the n-gram approach tended to favor sampled answers that were more verbose (this is a side effect of the math involved), whereas the dense embedding approach had no such bias. The more verbose answers were more likely to contain hallucinations (longer answers create more opportunities for hallucinations). But then why did the evals indicate that the n-gram approach was just as good as the dense embedding approach?

Fundamentally, the LLM judge was not grading answers in the way I intended. It was checking for the presence of correct information, but not for self-contradiction or the presence of incorrect information. This problem had always been there, but the use of the n-gram approach, with its bias towards longer answers, exposed it. This didn't necessarily mean the n-gram approach was worse (it was much faster than using dense embeddings), but it meant I no longer had reliable evidence that the n-gram approach preserved answer quality.

After all the confounding answers I’d received to board game questions, like the Grail Knights example above, I should have treated the LLM judge’s verdicts with more skepticism. Alas, the judge graded answers with the same misplaced confidence that Boardbarian brought to generating them.

When outputs are ultimately intended for human readers, an LLM judge can be a useful way to save time. However, its grading must be checked regularly against human expectations. In this case, I had assumed that the LLM judge was grading answers in a way that aligned with my expectations, but it was not. This misalignment led me to make a change that the evidence did not actually justify.

I don't use self-consistency in Boardbarian anymore (it does improve answer quality, but not enough to justify the cost), and I replaced the hand-rolled LLM judge with one built on DeepEval's G-Eval implementation. I don't actually know why it handles these cases better. It may be something inherent to G-Eval’s approach, or it may simply be that I wrote a better rubric the second time around. Before trusting it, I ran the same contradiction cases through it, and it caught what the old judge missed. I also make it a point to spot-check the results of the eval system on a regular basis. That's how I first noticed runaway generations: some answers were taking noticeably longer than others to come back. I already knew from experience that small models have a tendency to doom loop, so I suspected that was the cause, and the transcripts confirmed it. I added a dedicated metric to track it after that.

Did the judge's bias invalidate my earlier results? It may well have, and after the discovery I went back and reviewed my previous conclusions. Mostly, though, the question is moot: much of how Boardbarian works has changed since those experiments, and with a better judge and regular spot checks, the current state of the system is well supported by the current evals.

## Conclusion

At the end of the first post, I asked: how do I know any of this works? Here is what the evals currently show across the games *Warhammer Fantasy Battle*, *Munchkin*, and *Oathsworn*:

| Game | Unique Questions | Correctly Answered | Quote Validity | Non-Runaway Generations |
| -------- | -------- | -------- | -------- | -------- |
| *Warhammer Fantasy Battle* | 5 | 84% | 96% | 100% |
| *Munchkin*    | 20   | 96%   | 99% | 100% |
| *Oathsworn*   | 5   | 100%   | 98% | 100% |

The scores are averaged across all five runs of each question. The "Correctly Answered" column is the percentage of runs that the system answered correctly, as judged by the LLM judge. "Quote Validity" is the percentage of generated quotes that passed validation on the first attempt, before the repair loop ran. "Non-Runaway Generations" is the percentage of runs that did not result in a runaway generation.

Because I use the *Warhammer* and *Munchkin* questions during development, those results may be inflated by repeated development against the same questions; the held-out *Oathsworn* set is the closest thing here to an honest estimate. These are still small test sets, especially *Warhammer* and *Oathsworn*, so treat the percentages as directional rather than precise.

The honest caveat is that this is performance on my questions. I don't yet have much production data, and I don't yet run the evals against what I do have; when Boardbarian fails in production today, I mostly don't know it. Closing that gap is the obvious next step, and seeing how Boardbarian holds up against the questions actual players ask will doubtless lead to further refinements of the evals and the test cases. If you want to help with that, [Boardbarian](https://boardbarian.com) is live: bring it your gnarliest rules question and see if it holds up.

Looking back at the requirements for the eval system:

1. **Meaningful.** The test cases are real rules questions with hand-written expected answers, and the metrics track the failures that actually happen: wrong answers, hallucinated quotes, and runaway generations.

2. **Cheap.** The suite runs against small models, mostly on my homelab, which makes it affordable to run every question multiple times and compare many parameter configurations.

3. **Trustworthy.** This is the one where I got burned. The LLM judge was the only component of the system I had never evaluated, and it misled me for months. Trust in a judge isn't something you establish once; it has to be re-earned whenever the answers it grades change shape.

Boardbarian still fails. But no change ships without evidence, across the whole suite, that it fixed more than it broke. That discipline, more than any single design decision, is what the project runs on.

Next up is the system design: the infrastructure that runs these workflows and serves answers cheaply.
