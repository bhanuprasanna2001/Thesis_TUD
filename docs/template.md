You are my “paper sidecar” generator. Convert the paper into a vault note that lets me (1) recall the core in 5 minutes later, and (2) re-derive/implement the method without rereading.

# OUTPUT RULES (STRICT)
- Write in Obsidian-friendly Markdown with clear headings.
- Use $ for LaTeX math instead of [] or \[\] or () or \(\).
- Use LaTeX for all math. Keep equations copyable (no images of equations).
- When referring to content, always include (where possible): equation number, section number, and page number.
- Don’t handwave. If the paper skips steps, explicitly label it as “paper jump” and fill in missing algebra in a small derivation.
- If something is ambiguous, list the possibilities and what evidence supports each.
- Keep quotes ≤ 25 words. Prefer paraphrase.
- End with a checklist verifying you covered every required section below.

# CONSTRAINTS

- Extra constraint: treat me as new to the topic. Whenever you introduce a new concept (ELBO, score, posterior, schedule, etc.), add a 2–3 line mini-explainer and a tiny example.
- Extra constraint: after each section, add "Implementation translation" bullets: what tensors are needed, what shapes, what to precompute, and where numerical issues arise.
- Extra constraint: prioritize derivations. For every equation transformation, write:
  - starting expression,
  - identity used,
  - resulting expression,
  - what changed and why.
No skipped algebra unless trivial.

# CONTEXT
Paper title: <PASTE TITLE>
I care about: deep understanding, notation clarity, derivation flow, and implementation details.

# 0) TL;DR CARD (10 lines max)
- Citation:
- Problem (1–2 lines):
- Core idea (2–4 lines):
- Key contributions (≤3 bullets):
- Main results (numbers + dataset/metric):
- What’s actually new vs prior work:
- Assumptions / scope:
- When it fails / limitations:
- “If you remember only 3 things”:

# 1) GLOSSARY & NOTATION (NO EXCEPTIONS)
Create a table:
Symbol | Meaning | Shape/type | Where defined (sec/eq/page) | Notes
Include every symbol that appears in the method section.
Also include: dataset notation, time index, distributions, hyperparams, schedules, losses.

# 2) PROBLEM SETUP
- What is the data distribution and what is the modeling goal?
- What is the generative story (random variables + dependencies)?
- What is assumed known / fixed vs learned?

# 3) THE METHOD (PLAIN ENGLISH FIRST)
Explain the method as if to a smart peer in 8–12 sentences:
- What happens during training?
- What happens during sampling/inference?
- What is the “one trick” that makes it work?

# 4) MAIN EQUATIONS (THE CANONICAL SET)
Extract the “minimal sufficient” set of equations needed to reconstruct the method.
For each equation:
- Copy it in LaTeX
- Explain each term in one line
- State why it matters (what role it plays)
- Note where it is used next
Also create a short “equation dependency map”:
Eq A → used in Eq B → used in Algorithm / loss

# 5) DERIVATION MAP (NO BIG JUMPS)
Make a step-by-step arrow chain from assumptions → objective → simplified objective → final training loop.
Format like:
Step 1 (eq/sec/page): <what they start from>
→ Step 2: <what substitution/identity is applied>
→ Step 3: <what cancels/simplifies>
If the paper does variance reduction / reparameterization:
- Show the exact substitution
- Show what becomes constant / dropped and why
- Call out every inequality step (e.g., Jensen), what is concave/convex, and why it’s used

# 6) OBJECTIVE / LOSS (FINAL FORM + INTERPRETATION)
- Write the full objective(s) as used in experiments (final training loss).
- If there is a bound (ELBO/NLL surrogate), show how it decomposes.
- Explain what each term encourages.
- If they introduce “simple loss” or “weighted vs unweighted”: explain tradeoffs and what they actually trained with.

# 7) ALGORITHMS (TRAINING + SAMPLING)
Provide pseudocode for:
- Training loop
- Sampling/inference procedure
Include:
- Inputs/outputs
- Where randomness enters (what distributions are sampled)
- All hyperparameters used (timesteps, schedules, temperatures, etc.)
Also include a minimal PyTorch-like skeleton (no full codebase, just structure).

# 8) DESIGN CHOICES & ABLATIONS
Make a table:
Choice | Options tried | What changed | Effect on results | My takeaway
Examples: schedule choice, parameterization, variance choice, architecture, conditioning, weighting, timestep sampling.

# 9) IMPLEMENTATION & REPRODUCTION NOTES
Extract every detail needed to reproduce:
- Datasets + preprocessing
- Model architecture (layers, channels, attention, embeddings)
- Optimization (optimizer, LR, batch size, EMA, grad clipping, training steps)
- Regularization / tricks
- Compute used (GPUs/TPUs, runtime if given)
- Any “unstated defaults” you infer from context (label as inference)

Also include:
- “Gotchas & stability notes” (3–10 bullets)
- “Hyperparameters that matter most” (ranked)

# 10) RESULTS & EVALUATION
- Metrics used + what direction is better
- Main tables summarized (best numbers, datasets, conditions)
- Baselines and whether comparisons are fair (compute/data parity?)
- Failure modes or qualitative weaknesses (if any)
- If likelihood is reported, note exactly how computed/estimated

# 11) INTUITION & CONNECTIONS
- Intuition for why it works (mechanistic explanation, not vibes)
- Link to 3–7 related ideas/papers and how they connect (1–2 lines each)
- “What this paper secretly is” (e.g., score matching / variational inference / SDE view)

# 12) LIMITATIONS, ASSUMPTIONS, AND OPEN QUESTIONS
- Explicit limitations from the paper
- Implicit limitations you infer
- 5–10 open questions or “things I’d test next”
- What would break if assumptions are violated?

# 13) “STEAL THIS” SECTION (PORTABLE IDEAS)
List reusable techniques:
- Tricks, parameterizations, decompositions, schedules, objectives
For each: where used, why it helps, how to apply elsewhere

# 14) SELF-TEST (FOR LEARNING)
Create:
- 10 short questions (definitions, equations, steps)
- 5 medium questions (derive an equation, explain a design choice)
- 2 hard questions (extend/modify the method; predict outcome)
Include answers in collapsible format (or clearly separated).

# 15) FINAL CHECKLIST (MUST INCLUDE)
Before finishing, confirm:
- [ ] I listed the canonical equations and explained each
- [ ] I made a derivation map with no big jumps
- [ ] I gave sampling + training pseudocode
- [ ] I extracted all experimental/repro details
- [ ] I summarized results + ablations + limitations
- [ ] I wrote a complete notation glossary with shapes
- [ ] I included “gotchas” and “what to test next”