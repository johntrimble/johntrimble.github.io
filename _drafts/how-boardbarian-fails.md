---
layout: post
title: How Boardbarian Fails
excerpt: Wrong answers, hallucinated quotes, and doom loops. The evaluation system behind a small-model RAG assistant, and the LLM judge that lied to me for months.
date: 2026-07-17 09:57 -0700
media_subpath: /assets/img/posts/how-boardbarian-fails/
image:
    path: how_boardbarian_fails.png
---

Consider this question about *Warhammer Fantasy Battle 5th Edition*: "Do Grail Knights have to take a Break test when they lose a round of combat?" [Boardbarian](https://boardbarian.com) might respond with:

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

And if you think this is just a retrieval failure, consider yet another gem of an answer to the same question:

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

Incredible. Both of these are real answers Boardbarian produced at some point during its development, though not necessarily answers the current version would give. Most of my time working on Boardbarian has been spent debugging failures like these.

## Requirements

Failures like these are why most of my effort has gone into evaluations. Debugging them one at a time by hand doesn't scale, as we'll get to shortly, so early on I invested in an automated eval system, with these requirements:

1. **Meaningful.** It has to catch real failures on real questions, not benchmark trivia. Real questions carry all the game-specific vocabulary, meta-rules, and multi-hop retrieval the first post described.
2. **Cheap.** I run lots of experiments, mostly on my homelab, so a full eval run has to be affordable enough to repeat constantly.
3. **Trustworthy.** When the evals say one approach beats another, that verdict has to be one I can act on. (Foreshadowing: this one gets its own section later.)


## Whack-a-mole

When a failure shows up, the temptation is to fix it directly: adjust the prompt, tweak a sampling parameter, rerun the failing question, and celebrate when it passes. The problem is that every one of these knobs is connected to everything else. Tweaking sampling parameters to resolve runaway generations can degrade quoting accuracy. Rewording the prompt to produce more accurate answers can increase the tendency toward runaway generations. Fix the Grail Knights question, and something you didn't think to recheck quietly breaks.

Manual spot-checking can only tell you whether a change fixed the case you were staring at. It says nothing about what the change did to every other question. After enough rounds of this, the conclusion was hard to avoid: you can't fix what you can't measure.


## Measuring failure

Boardbarian fails in three distinct ways, and the eval system has a metric for each:

1. **Wrong answers.** The headline metric is correctness: does the system's answer agree with a hand-written expected answer? An LLM-judge grades this, using an implementation of the approach from [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (Liu et al., 2023)](https://arxiv.org/abs/2303.16634), built on [DeepEval](https://github.com/confident-ai/deepeval).

2. **Fabricated evidence.** Every answer must support itself with quotes from the rulebook, and as described in the first post, quotes are validated at runtime with fuzzy matching. The evals track how often the model hallucinates quotes. Even when the repair loop catches them, a rising hallucination rate means more retries, which means slower and more expensive answers.

3. **Never finishing.** Runaway generations, where the model gets stuck in a loop and generates until it hits the token limit. I promised the doom loop story back in the previous post: this is where doom loops get counted.

The test cases are real rules questions paired with hand-written expected answers. Because the system is stochastic, each question runs multiple times per experiment. A question that passes three times out of five is a very different beast from one that passes five out of five, and a single run can't tell you which you have. All of this runs against small models, mostly on my homelab, which is what keeps a full suite run cheap enough to repeat for every experiment in the next section.


## Using evals to make design decisions

Back when I started, there was a lot of discussion about how to do RAG properly: how big chunks should be, whether chunking should follow the semantic structure of the document, which embedding model to use, whether vector search alone is enough. Most of the advice on offer was argued from first principles, or from vibes. Coming from a bit of an ML background, I saw these as hyperparameters, and there is a boring, well-worn method for choosing hyperparameters: enumerate the options, search over them, and measure. The eval suite is what made that possible. Some results:

**Chunking.** After comparing a variety of chunk sizes, and whether or not to use parent-child chunking, it turns out using parent-child chunking with a parent chunk size of roughly 500 tokens and a child chunk size of roughly 125 tokens works best. I say roughly, because some considerations are taken to not split at an inconvenient place, like in the middle of a sentence.

**Embeddings.** Could I get away with a smaller, cheaper embedding model? Yes. I replaced [jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) with [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), which is smaller, faster, and cheaper, with no measurable impact on the evals.

**Hybrid search.** Is hybrid search actually better than plain vector search? It turns out the answer is yes, though the evals originally indicated it was not. I was actually quite surprised by this, so I investigated further and found that it was a LangChain bug: [langchain-ai/langchain-postgres#288](https://github.com/langchain-ai/langchain-postgres/issues/288). After working around the bug, the evals showed that hybrid search was indeed better than vector search alone, at least for this use case.

**Sampling parameters.** The main defense against doom loops is tuning sampling parameters, but those parameters also affect answer quality, so this was a search for a configuration that suppresses runaway generations without degrading correctness or inhibiting accurate quoting from the context. The evals made it possible to search over a grid of sampling parameters and find a configuration that worked well. I've had to do this multiple times as I switched models and providers, and each time the evals made it possible to find a good configuration quickly.

**Self-consistency.** The most involved experiments compared different self-consistency approaches. The evals crowned a winner here too. I'll come back to why that turned out to be a problem.

**Prompting.** The evals have been invaluable for prompt tuning. I can make a change to the prompt, run the evals, and see if it improved or degraded performance. This is especially useful as tuning the prompt to better handle one sort of question can easily degrade performance on another sort of question, or even push the model into a doom loop. The evals make it possible to see the impact of prompt changes across the whole suite of questions. You can even use the evals to automate prompt tuning, though I haven't done that myself.

## Evals as insurance

Right as I was pushing everything into production, my inference provider at the time, Atlas Cloud, dropped support for the model I was using. (The architectural side of that story, the provider fallbacks and circuit breakers, comes in the next post.) The eval suite is what turned it from a crisis into a chore. As a stopgap, I swapped in Qwen3.6 35B A3B, which I judged to be the most reliable replacement available in the interim. Since the prompts and workflow were built around Qwen3 30B A3B Instruct, I disabled thinking on the new model to make it behave more like the model it was replacing, reran the evals to validate that it was a safe substitution, and reran the sampling parameter grid search to retune for the new model. That last part was not optional: different providers support different sampling parameters, so a configuration tuned for one provider may not even be expressible on another.

I also keep fallbacks on Qwen3 30B A3B Instruct through two other providers, retuned for the sampling parameters those providers actually support. It's an awkward arrangement: the model the system was designed around now serves as the fallback, while the primary is a newer model doing an impression of it. The evals are what let me be comfortable with that.

This is the quieter payoff of an eval system. It isn't just for choosing between designs during development. It's what lets you absorb changes you didn't choose, like a discontinued model or a new provider, and know, rather than hope, that the system still works.

## When the judge lies

I have used an LLM-judge to grade answers from the very beginning of the project. By the time I got to the self-consistency experiments, it had earned a fair amount of trust: across the chunking, embedding, and search comparisons, its verdicts had generally matched my own reading of the transcripts.

When evaluating self-consistency approaches, I was comparing the approach I was using with dense embeddings to one from Amazon, described in [Lightweight reranking for language model generations (Jain et al., 2023)](https://arxiv.org/abs/2307.06857), that used sparse n-gram vectors. According to my eval results, which used a hand-rolled LLM-judge, the Amazon approach was basically as effective while being cheaper and faster. Consequently, I switched my approach.

Much later, and sadly I do mean much later, I discovered that the judge had been giving too much credit to answers produced by the n-gram approach. In reality, the answers produced were more verbose and had a tendency to contain hallucinations. The reason the judge misgraded the answers was that it was only checking if the answer contained the correct information, not whether it was concise or contained additional inaccurate information.

The lesson here is that having an eval system in place is essential, but the eval system itself should be evaluated. This means having a human in the loop to spot check the results of the eval system, and to ensure that the eval system is grading answers in a way that aligns with the goals of the project. I learned this lesson the hard way.

I don't use self-consistency in Boardbarian anymore (it does improve answer quality, but not enough to justify the cost), and I replaced the hand-rolled LLM-judge with one built on DeepEval. However, I do make it a point to spot check the results of the eval system on a regular basis. This spot checking has helped me uncover other issues, such as runaway generations, which I ended up adding a metric to track.

Did the judge's bias invalidate my earlier results? It may well have, and after the discovery I went back and reviewed my previous conclusions. Mostly, though, the question is moot: much of how Boardbarian works has changed since those experiments, and with a better judge and regular spot checks, the current state of the system is well supported by the current evals.

## Conclusion

At the end of the first post, I asked: how do I know any of this works? Here is the answer the evals give: on the test data, Boardbarian answers correctly about 91% of the time. Because every question runs multiple times, that number reflects how the system usually behaves, not how it behaved once. The honest caveat is that this is performance on my questions. I don't yet have much production data, and I don't yet run the evals against what I do have; when Boardbarian fails in production today, I mostly don't know it. Closing that gap is the obvious next step, and seeing how Boardbarian holds up against the questions actual players ask will doubtless lead to further refinements of the evals and the test cases. If you want to help with that, [Boardbarian](https://boardbarian.com) is live: bring it your gnarliest rules question and see if it holds up.

Looking back at the requirements for the eval system:

1. **Meaningful.** The test cases are real rules questions with hand-written expected answers, and the metrics track the failures that actually happen: wrong answers, hallucinated quotes, and runaway generations.

2. **Cheap.** The suite runs against small models, mostly on my homelab, which makes it affordable to run every question many times and to rerun everything for each experiment. Cheap evals are what made grid search practical at all.

3. **Trustworthy.** This is the one where I got burned. The LLM-judge was the only component of the system I had never evaluated, and it misled me for months. Trust in a judge isn't something you establish once; it has to be re-earned whenever the answers it grades change shape.

Boardbarian still fails. But no change ships without proving, across the whole suite of questions, that it fixed more than it broke. That discipline, more than any single design decision, is what the project runs on.

Next up is the system design: the infrastructure that runs these workflows and serves answers cheaply.
