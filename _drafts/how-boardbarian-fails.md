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

At the end of the [previous post]({% post_url 2026-08-03-how-boardbarian-thinks %}), I asked how I know any of this works. The honest answer starts with failure: getting [Boardbarian](https://boardbarian.com) to a working system meant spending most of my time staring at wrong, sometimes bewildering, answers. This post is about the eval system I built to make sense of that: not just to catch failures, but to tell whether a given change actually helped. The system worked. Except for the LLM judge, which lied to me for months.

Let's return to our example from the last post, from the game *Warhammer Fantasy Battle 5th Edition*: "Do Grail Knights have to take a Break test when they lose a round of combat?" As a refresher, the answer is yes: Grail Knights are immune to psychology tests, but the rulebook explicitly says that a Break test is not a psychology test. Here is Boardbarian making the natural but incorrect assumption that Break tests fall under the umbrella of psychology:

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

Even when Boardbarian has all the information it needs, it can still fail. In the following answer, it quotes the rule stating that a Break test is not a psychology test, reasons through the distinction correctly, and still reaches the wrong conclusion:

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

1. **Meaningful.** It has to catch real failures on real questions. Real questions carry all the game-specific vocabulary, meta-rules, and multi-hop retrieval the first post described.
2. **Cheap.** I run lots of experiments, mostly on my homelab, so a full eval run has to be affordable enough to repeat constantly.
3. **Trustworthy.** When the evals say one approach beats another, that verdict has to be one I can act on.


## Whack-a-mole

When a failure shows up, the temptation is to fix it directly: adjust the prompt, change how the model generates its answer, rerun the failing question, and call it fixed. The problem is that every one of these knobs is connected to everything else. A change that suppresses runaway generations can degrade quoting accuracy. Rewording the prompt to produce more accurate answers can make runaway generations more likely. Fix the Grail Knights question, and something you didn't think to recheck quietly breaks.

Spot-checking is valuable. It's how you notice something is wrong in the first place, and it stays useful even after you've built an automated system, as I'll get to later. But it scales poorly on its own: checking whether a change fixed one issue doesn't tell you if it caused another. To iterate quickly, you need something spot-checking alone can't give you: automated, broad, repeatable coverage.


## Measuring failure

Boardbarian fails in three recurring ways, and the eval system has a metric for each:

1. **Wrong answers.** The headline metric is correctness: does the system's answer agree with a hand-written expected answer? Correctness is graded by an LLM judge built on [DeepEval](https://github.com/confident-ai/deepeval)'s implementation of [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (Liu et al., 2023)](https://arxiv.org/abs/2303.16634), which replaced an earlier hand-rolled judge (more on that swap later). Since making the switch, and moving to newer models along the way, I've re-run or redone nearly every experiment described in this post under the current judge.

2. **Fabricated evidence.** Every answer must support itself with quotes from the rulebook, and as described in the first post, quotes are validated at runtime with fuzzy matching. The evals track the percentage of individual generated quotes that pass validation on the first attempt, before the repair loop runs. Even when the repair loop catches an invalid quote, a falling first-pass validity rate means more retries, which means slower and more expensive answers.

3. **Never finishing.** Runaway generations, where the model gets stuck in a loop and generates until it hits the token limit. I first noticed these because some answers were taking noticeably longer than others to come back. Small models have a tendency to doom loop, so I suspected that was the cause, and the transcripts confirmed it. I added a dedicated metric to track it after that.

The test cases are real rules questions paired with hand-written expected answers. The suite currently contains 30 questions across three games: five for *Warhammer Fantasy Battle*, twenty for *Munchkin*, and five for *Oathsworn*. I use the *Warhammer* and *Munchkin* questions during development. *Oathsworn* serves as a small holdout set that did not influence development of the version evaluated here, although it will need to be refreshed as its results begin to influence future changes.

Because the system is stochastic, some questions produce more variable answers than others. I therefore run each question five times per experiment to measure how reliably the system answers it correctly. All of this runs against small models, mostly on my homelab, which is what keeps a full suite run cheap enough to repeat for every experiment in the next section.


## Evals for design decisions

Once the suite existed, I could treat design choices as experiments. Back when I started, there was a lot of discussion about how to do RAG properly: how much text to search at once and how to split documents into passages (chunking), whether to preserve the structure of the document, which model to use for comparing passages by meaning, and whether searching by meaning alone (vector search) was enough.

Most of the advice on offer rested on theoretical arguments or anecdotal experience rather than comparative measurements. These were effectively hyperparameters, so I treated them the usual way: enumerate the options, search over their combinations where necessary, and measure the results.

The point here is not that these settings generalize to other RAG systems, but that the suite let me test them against Boardbarian’s actual workload.

A few of the results were quick wins. Searching over small passages while returning the larger surrounding passage performed better than searching over passages of a single fixed size. Replacing the model used to compare passages by meaning (the embedding model) with a smaller, cheaper alternative showed no measurable loss in accuracy. Prompt tuning stayed useful throughout: a change that fixed one class of question could easily degrade another or push the model into a doom loop. The evals made those regressions visible across the suite.

Three other results are worth a closer look.

**Hybrid search.** Is combining search by meaning with search for exact wording actually better than searching by meaning alone? It turns out the answer is yes, though the evals originally indicated otherwise. I was surprised by this, so I investigated and found a LangChain bug: [langchain-ai/langchain-postgres#288](https://github.com/langchain-ai/langchain-postgres/issues/288). After working around it, the evals showed that the combined approach performed better, at least for this use case.

**Sampling parameters.** When a model generates text, it assigns probabilities to the possible next tokens, roughly words or pieces of words. Sampling parameters control how the model chooses among them. These settings are the main defense against doom loops, but they also affect answer quality, so I needed a configuration that suppressed runaway generations without degrading correctness or accurate quoting. The evals let me search over combinations and measure which one worked best. I repeated that search whenever I switched models or inference providers (the services running the models).

**Self-consistency.** The most involved experiments compared ways of generating several answers to the same question and selecting the consensus answer. The evals crowned a winner here too. I'll come back to why that turned out to be a problem.

## When the judge lies

When I started the project, one of the first things I did was hand-roll an LLM judge for grading rule answers, and validate it against 21 *Munchkin* question-and-answer pairs I'd graded myself. It earned a fair amount of trust by helping me evaluate chunking, embedding, and search approaches. I became quite confident in how well it worked. As it turns out, I was overdue for a lesson in humility.

Boardbarian used to use self-consistency, a technique first described in [Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2023)](https://arxiv.org/abs/2203.11171). Selecting from several sampled answers can reduce the impact of any single bad answer and improve overall accuracy. For my use case, the challenge was determining which answer represented the consensus, since open-ended answers could vary in wording while conveying the same meaning. I compared several methods using the eval system and my custom LLM judge.

Originally, I'd settled on an approach that compared answers by meaning using dense embeddings. I later compared it to an approach from Amazon, described in [Lightweight reranking for language model generations (Jain et al., 2023)](https://arxiv.org/abs/2307.06857), that compared shared words and short phrases using sparse n-gram vectors. The evals indicated that the Amazon approach was basically as effective while being cheaper and faster, so I switched.

Much later, and sadly I do mean much later, I began to feel, while using Boardbarian, that the answers were often self-contradictory. For example, for the game *Munchkin*, I'd ask the question: "When can I play a Hireling?" The answer I would expect is:

> You can play a Hireling at any time. However, you can only have one Hireling in play at a time.

What I'd sometimes get is something like the following:

> You can play a Hireling whenever it is legal to play a Class or Race card, as long as you have an appropriate card to attach the Hireling to. Hirelings can be found in Door decks or Treasures, and you can play them at any time, even in combat, as long as you have only one Hireling in play at a time.

The LLM judge would mark both answers as correct, even though the second answer is verbose and contradicts itself. It says a Hireling can be played "whenever it is legal to play a Class or Race card," which is incorrect, but then later says "you can play them at any time," which is correct.

After inspecting the sampled answers used in self-consistency, I found that the n-gram approach tended to favor more verbose answers. It scores answers based on how many words and short phrases they share with the other samples. Longer answers have more opportunities to contain matching language and earn a higher score, whereas the dense embedding approach had no such bias. The more verbose answers were also more likely to contain incorrect claims, in part because they made more claims. But then why did the evals indicate that the n-gram approach was just as good as the dense embedding approach?

Fundamentally, the LLM judge was not grading answers in the way I intended. It was checking for the presence of correct information, but not for self-contradiction or the presence of incorrect information. This blind spot had always been there. The validation set I built when I first stood up the judge never tested for it either: none of the examples in it were self-contradictory or padded with extra false claims, so there was nothing there to expose the gap. The use of the n-gram approach, which favors longer answers, exposed the problem. I would have caught it sooner had I been spot-checking the results regularly. This didn't necessarily mean the n-gram approach was worse (it was much faster than using dense embeddings), but it meant I no longer had reliable evidence that the n-gram approach preserved answer quality.

After all the confounding answers I’d received to board game questions, like the Grail Knights example above, I should have treated the LLM judge’s verdicts with more skepticism. Alas, the judge graded answers with the same misplaced confidence that Boardbarian brought to generating them.

I don't use self-consistency in Boardbarian anymore (it does improve answer quality, but not enough to justify the cost), and I replaced the hand-rolled LLM judge with one built on DeepEval's G-Eval implementation. I retested it against the same contradiction cases, and it rejected them correctly. I also make it a point to spot-check the results of the eval system regularly.

Did the judge's bias invalidate my earlier results? It may well have, and after the discovery I went back and reviewed my previous conclusions. Mostly, though, the question is moot: much of how Boardbarian works has changed since those experiments. The larger lesson is that an LLM judge buys you breadth and speed: more test cases, checked faster, than I could ever manage by hand. It doesn't get you out of spot-checking the results yourself. That part doesn't go away.

## Evals as insurance

Evals were not only a development tool; they also gave me a migration path when a production dependency disappeared.

While I was pushing Boardbarian into production, Atlas Cloud, my inference provider at the time, dropped support for Qwen3 30B A3B Instruct, the model I was using. I discovered the change when production API calls began failing. By then, I had spent months making and testing changes against that model. It was not an ideal situation, but the evals let me evaluate replacements with some confidence that they would actually work.

I swapped in Qwen3.6 35B A3B, which appeared to be the most reliable replacement available. Because I was replacing an instruct model, I disabled thinking on the new model to make its behavior more closely resemble the old one. I then reran the search over combinations of sampling parameters and the full eval suite before putting the new model into service.

The sudden deprecation was annoying, but the evals let me find and validate a replacement quickly.

## Conclusion

At the end of the first post, I asked: how do I know any of this works? Here is what the evals currently show across the games *Warhammer Fantasy Battle*, *Munchkin*, and *Oathsworn*:

| Game | Questions | Correct Runs | First-Pass Quotes | Runaway-Free Runs |
| -------- | -------- | -------- | -------- | -------- |
| *Warhammer Fantasy Battle* | 5 | 21/25 | 96% | 25/25 |
| *Munchkin*    | 20   | 96/100   | 99% | 100/100 |
| *Oathsworn*   | 5   | 25/25   | 98% | 25/25 |

Each question runs five times. "Correct Runs" and "Runaway-Free Runs" show how many of those runs, out of the total for that game, met the bar (Warhammer's 21/25, for instance, means 5 questions run 5 times each, with 21 of those 25 runs judged correct). "First-Pass Quotes" is the percentage of individual generated quotes that passed validation before the repair loop ran.

Because I use the *Warhammer* and *Munchkin* questions during development, those results may be inflated by repeated development against the same questions; the held-out *Oathsworn* set is the closest thing here to an honest estimate. These are still small test sets, especially *Warhammer* and *Oathsworn*, so treat the percentages as directional rather than precise.

The honest caveat is that this is performance on my questions. I don't yet have much production data, and I don't yet run the evals against what I do have; when Boardbarian fails in production today, I mostly don't know it. Closing that gap is the obvious next step, and seeing how Boardbarian holds up against the questions actual players ask will doubtless lead to further refinements of the evals and the test cases. If you want to help with that, [Boardbarian](https://boardbarian.com) is live: bring it your gnarliest rules question and see if it holds up.

Looking back at the requirements for the eval system:

1. **Meaningful.** The test cases are real rules questions with hand-written expected answers, and the metrics track the failures that actually happen: wrong answers, hallucinated quotes, and runaway generations.

2. **Cheap.** The suite runs against small models on my own hardware, so a full run costs me electricity, not API calls. That's what makes it affordable to run every question multiple times and compare many parameter configurations.

3. **Trustworthy.** This is the one where I got burned. I had validated the original LLM judge, but only against answers that lacked the failure mode that ultimately exposed it. It misled me for months because its test set did not evolve as the answers it graded changed shape. Trust in a judge is not something you establish once; its own evaluations need to keep pace with the system it evaluates.

Boardbarian still fails. But no change ships without evidence, across the whole suite, that it fixed more than it broke. That discipline, more than any single design decision, is what the project runs on.

Next up is the system design: the infrastructure that runs these workflows and serves answers cheaply.
