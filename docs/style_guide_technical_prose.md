# A Style Guide for Technical Prose

Derived from line-by-line editing of an NSF proposal in computational science,
2026-09-04 through 2026-09-07. Every rule below was learned the same way: a draft
sentence was rejected, and the accepted replacement showed what the objection was.
The rules are therefore descriptive of one working style rather than universal, but
they generalize to proposals, papers, and technical reports in any quantitative field.

Two things make this list worth keeping. First, roughly a third of the rules are
*counter-rules*, cases where the correct action is to leave text alone; those are the
ones an eager editor gets wrong most often. Second, several rules describe the editing
process rather than the prose, and those transfer across projects more reliably than
any wording preference.

## 1. Sentence construction

### 1.1 No reduced relative clauses with the pronoun dropped
> the overhangs a partially completed part exhibits

The reader parses "overhangs a partially completed part" as verb plus object, hits
"exhibits", and reparses. The risk is highest when the head noun can also be read as
a verb: overhangs, shifts, bounds, fields, results, rates, states, forms.
Repair with a past participle, or restore the pronoun when the active voice is wanted:
> the overhangs exhibited by a partially completed part
> the overhangs that a partially completed part exhibits

This construction is easy to spot in someone else's draft and almost invisible in
your own. Check your replacements, not only the original.

### 1.2 No pseudo-clefts, in either direction
> What these methods inherit is the cut cell itself
> Building the integration domain from whole elements is what makes the method suitable

Delete the copular scaffolding and let the verb be the verb:
> These methods inherit the cut cell itself
> Whole-element integration suits the method directly

This construction is also the default repair people reach for when fixing a vague
demonstrative, so it reappears exactly where something else was just fixed.

### 1.3 Parallel active clauses for consequences; parallel noun phrases for objects
When listing two related technical consequences, give each its own subject and verb
rather than hanging nominalized fragments off one governing verb:
> What these methods inherit is the cut cell: quadrature on arbitrarily cut elements
> and the conditioning degradation a small cut produces

becomes
> cut-cell integration requires quadrature on arbitrarily cut elements, and small
> cuts lead to degraded conditioning

**The exception matters as much as the rule.** When the items genuinely are the direct
objects of a single verb, noun phrases are the parallel form and embedded clauses are
the clumsy one:
> determine which surfaces are overhangs, how the thermal history is distributed, and
> how much post-processing is required                                    (clumsy)
> determine overhangs, the distribution of thermal history, and the degree of
> post-processing required                                                (correct)

Diagnose which case you have before applying the rule.

### 1.4 "Does not require X", never "requires no X"
> the method requires no mesh generation at any increment          (informal)
> the method does not require mesh generation at any increment     (correct)
> no mesh generation is required at any increment                  (also correct)

The passive alternative is fine. Only the "requires no X" / "needs no X" form is out.
Plain possessive negation is unaffected: "has no standard stability theory" is fine.

### 1.5 Name the relation of a trailing list with "including"
> The first aim delivers the representation, the training procedures that make it
> tractable, and the error analysis certifying it

The reader cannot tell whether items two and three are further objects of "delivers"
or appositives to the first. Insert "including" and the parse is fixed. Whenever a
sentence's object is followed by additional related items, state the relation.

### 1.6 Group list items by context, and lead each item with its context
Items sharing a governing regime go together, not in whatever order they occurred to
you, and the regime is stated before the item so the reader knows which case applies
while parsing it:
> Layer time and inter-layer cooling in process A determine ...     (context buried)
> In process A, the layer time and inter-layer cooling determine ... (context leads)

Reordering a list is usually a bigger improvement than rewording it.

### 1.7 No colon-plus-list shorthand for a sequence
> X assumes a fixed geometry before Y begins: A, then B, then C.

Write it as prose: "X assumes A before Y begins, moving from A to B and then to C."

### 1.8 No em-dashes
A house convention rather than a general rule, but an absolute one where it applies,
and worth settling before drafting rather than sweeping for afterward. Use a comma, a
semicolon, a colon, or two sentences. Confirm the convention per project; some venues
are indifferent and some authors want them.

### 1.9 Attach a relative clause to the noun it modifies
> differentiable with respect to geometry, material fields, and boundary data, which we
> cross-check against finite differences

"which" lands on "boundary data" and means the derivatives. Move the clause, or restate
the antecedent: "..., and we cross-check these derivatives against finite differences."
This survives style passes because every word in it is correct.

### 1.10 No sentence fragments, and a semicolon needs a full clause on both sides
> ... which restricts the search to two or three parameters; a restriction which does
> not apply to gradient-based optimization.

What follows the semicolon is a noun phrase, not an independent clause. Either promote
it to a clause, or demote the semicolon to a comma and let the phrase be an appositive:
> ..., a restriction that does not apply to gradient-based optimization.

Note the same edit fixes "a restriction *which*" to "a restriction *that*": a
restrictive clause takes "that".

### 1.11 Do not elide a verb across an intervening "and"
A second clause that borrows its verb from the first reads as a run-on once anything
sits between them. Repeat the verb, or split the sentence.

### 1.12 Break chains of stacked "of" phrases
> optimization of statistics of a quantity of interest

Three "of" phrases in a row force the reader to hold two incomplete noun phrases open.
Convert one link to a different relation, or name the compound directly: "optimizing
the statistics of a quantity of interest", or "optimizing a risk measure of the
quantity of interest". Two "of" phrases are usually fine; three are a signal.

### 1.13 Keep a prepositional phrase next to what it modifies
> this map enables converting the process-parameter selection into gradient-based
> optimization from an expensive search over simulations

"from an expensive search" modifies the conversion and has been stranded seven words
past it, after "into". A converts-from-X-into-Y sentence must keep the from and the
into adjacent and in that order, or be recast so only one of them is needed.

### 1.14 Put the real subject first; do not front it behind a participial, absolute, or gerund-phrase modifier
> With its nonlinear features frozen, a trained INR defines a reduced family of
> nearby surfaces through perturbations of its final layer.
> Doing this from one trained network raises three difficulties.
> Our key idea is to freeze the INR's nonlinear features and treat only its final
> layer as free, which makes ...

Each buries the true subject (the network, the propagation task, "we") behind an
introductory phrase the reader has to hold before the sentence's real actor shows up.
Put the subject first and demote the modifier:
> A trained INR, with its nonlinear features frozen, defines a reduced family of
> nearby surfaces through perturbations of its final layer.
> Performing uncertainty propagation using INRs raises three challenges.
> We freeze the INR's nonlinear features and treat only its final layer as free,
> which makes ...

If the modifier is kept rather than cut, set it off with commas on **both** sides
once it follows the subject; a single trailing comma turns it into a dangling
fragment between subject and verb.

This is distinct from 1.2 (pseudo-clefts): the defect here is a fronted modifier
delaying the subject, not a copular "is / is what" scaffolding, though both read as
inversion and both are fixed the same way: find the real subject and verb, and lead
with them.

## 2. Verbs

### 2.1 Plain verbs, never figurative ones
Watchlist: **consume, inherit, carry, drive, feed, absorb, supply, unlock, leverage,
harness, bridge, marry, capture** (when not literal), **lock, stall, hinge on.**
> both are consumed by the same solver     ->   both are used by the same solver

The same prohibition covers extended figures. "A domain that grows while the physics
runs on it" became "the changing domain". "The geometry acquires a time derivative and
the build acquires a calculus" was rejected outright as reading like machine-written
text; the repair was to name the mathematical object directly: "its time derivative,
the rate at which the domain grows, is well defined at every instant of the build."
When a sentence claims something *acquires* an abstract property, replace it with a
direct statement of what is actually well defined or computable.

### 2.2 Transformation metaphors become plain statements of the operation
> The network supplies that distance vector
>   -> The network is used to compute the distance vector
> Neural distance fields drive the solver directly
>   -> Neural distance fields can be used for the solver directly
> Adding time as a coordinate turns thousands of states into one network
>   -> Adding time as a coordinate enables representing thousands of states in a
>      single network
> Composing A with B provides that gradient
>   -> That gradient is computed by composing A with B

A passive is perfectly acceptable here. The figurative active verb is the defect, not
the voice.

### 2.3 Not weak, not colloquial, not subfield jargon, not economic
- **Weak:** prefer *determine* to *set*, *requires* to *needs*, *is* to *collects*.
- **Colloquial:** "hot material lands" became "hot material gets deposited".
- **Subfield jargon:** "leaves green material" became "leaves material uncured". A
  term that is precise inside a subfield can still be unreadable one field over. In
  reviewed work, assume at least one reader is outside your specialty.
- **Economic:** *affordable, cheap, expensive, pays for itself* read as loose in a
  technical context. Write **computationally tractable**, *feasible*, or a
  stated complexity. Reserve cost words for actual money. The exception is a genuine
  cost analysis, where "pays" is the established idiom of the crossover argument.

### 2.4 Name the operation as the subject, not the abstract object
When an abstract mathematical object is being acted on, make the *act* the subject:
> That gradient turns parameter selection into ...
>   -> Computing this gradient turns parameter selection into ...

Corollary, and the more common failure: an abstract object should not be the agent of
an action only a person or a program performs.
> the in-situ record and metrology update the posterior
>   -> the posterior is updated from the in-situ record and metrology
> the eigenmodes parameterize the realizable edits
>   -> the realizable edits are parameterized by the eigenmodes

Note that 2.4 and 1.2 pull in opposite directions if applied carelessly: making the
operation the subject must not produce "X is what does Y".

### 2.5 Name the computational task, not the phenomenon
> Its physics is attenuation of the projected light coupled with cure kinetics.
>   -> Simulating the process requires solving optical transport equations for the
>      attenuation of the projected light coupled with polymer cure kinetics.

Two moves in one. A copular "its physics is X" sentence parks the content in a bare
noun phrase; an active clause names what must be computed. And in a computational
proposal, the salient fact about a physical process is the system of equations it
forces you to solve, not a label for the phenomenon. Whenever tempted to write
"the physics is ...", write "simulating ... requires solving ..." instead.

### 2.6 Predicate adjective on the object, not adverb on the defining verb
> defines the growth rate continuously everywhere    (adverb modifies the defining)
> defines the growth rate to be continuous everywhere (adjective states the property)

### 2.7 Future tense for work not yet done
> The first aim delivers the representation ...     (asserts a completed result)
> The first aim focuses on developing the representation ...
> We will construct ..., We will establish ...

In a document proposing work, a present-tense achievement verb claims the award has
already produced something. This extends to any section describing planned activity,
including assessment, dissemination, and outreach.

### 2.8 Cut empty intensifier adverbs
> the trade-off that this gradient quantifies directly
> the trade-off that this gradient quantifies

*Directly, exactly, precisely, fully, simply, readily, naturally, effectively* almost
never survive a challenge. Either the verb is true without the adverb, in which case
the adverb is padding, or the adverb is carrying a claim the verb does not support, in
which case it should be argued rather than asserted.
Keep the adverb only when it draws a contrast the reader needs: "computed directly
rather than by finite differences" earns it.

## 3. Nouns, reference, and register

### 3.1 Precise technical nouns, not placeholders
> the settings that produced it   ->   the process parameters that produce it

Placeholder watchlist: **settings, things, factors, aspects, machinery, framework,
pipeline, approach, procedure, elements, components, capabilities.**

### 3.2 Repeat the noun rather than pronoun across a sentence boundary
> that history   ->   the geometry history

The extra word or two buys precision. Two consecutive sentences opening with "It" is
always a defect.
**The counter-rule:** a demonstrative pointing at a concrete, immediately preceding,
*named* subject is fine. "This approach also makes ..." is acceptable when the approach
was just named. Do not over-apply the referent test to demonstratives that have a real
antecedent one sentence back.

### 3.3 Narrow a vague noun minimally, not maximally
> makes the whole framework brittle
> -> makes the simulation framework brittle       (accepted: light touch)
> -> makes per-layer remeshing brittle            (rejected: over-specified)

When flagging a vague noun, prefer dropping or replacing the vague qualifier over
rewriting the whole noun phrase into the most specific term available.

### 3.4 Use the full standard term
> cure kinetics        ->   polymer cure kinetics
> under a projected image ->  under projected UV or visible light

Add the material or physical qualifier rather than changing the head noun, and name
the actual physical actuator. When shortening a term saves only a word or two, use the
complete conventional name. This does not conflict with 4.3, which is about adding
redundant *examples*; this is about the complete *name* of a thing.

### 3.5 Mark scope explicitly, and collapse repeated pronoun objects
> The first contribution is ...     ->  The first contribution of this work is ...
> We will construct it, train it, and establish the conditions under which it supports
>   a convergent solve
>   -> In this work we will construct and train this representation, and establish the
>      conditions under which the representation supports a convergent solve

Two moves: add "of this work" / "In this work" rather than leaving scope to context,
and when a verb list repeats a pronoun object, collapse to one shared explicit noun
and name the noun again later instead of a third "it".

### 3.6 Put "for example" inside an illustrative parenthetical
> (residual stress, warping, dimensional error)
> (for example, residual stress, warping, dimensional error)

when the list is not exhaustive.

### 3.7 Word preferences
*Challenge*, not *problem*, for the thing the work addresses in a framing sentence.

### 3.8 Audience governs register, section by section
A symbol that is perfectly clear in a technical section becomes an obstacle in a
section written for a different reader. In a proposal that means no notation in
Broader Impacts, Management Plan, or budget justification; in a paper it means the
abstract and often the introduction; in a report it means the executive summary.
> representing the part as $f_\theta(x,t)$ removes the meshing step
> -> Representing the part using a single neural object removes the meshing step

The same object can carry its symbol in the technical sections and a plain noun phrase
elsewhere. That is not an inconsistency to fix. Sweep for stray notation whenever
editing a section aimed at a different reader.

### 3.9 Fix the spelling convention once, and hold it
US or UK English, chosen deliberately and applied throughout. Mixed conventions read as
carelessness to a reviewer even when every individual spelling is defensible. The
common offenders in technical writing are *focuses/focusses*, *modeling/modelling*,
*behavior/behaviour*, *analyze/analyse*, and *center/centre*.

## 4. Sentences and paragraphs that should be deleted

The single most useful test in this guide: read a sentence and ask **"What is that?"**
If the sentence names an abstract subject, applies a vague predicate, and promises a
specific that has not yet appeared, delete it.

### 4.1 Cut vague hook sentences, openers included
> Current practice stalls on the geometry in a specific, diagnosable way.
> This places learning at exactly one point in the pipeline.

Neither names a mechanism. Both gesture at one about to be explained.
**A vague opener is not rescued by the paragraph delivering on it later.** The
temptation is to keep such a sentence as a topic sentence on the grounds that the
specific arrives three sentences on. It does not survive that defense: the later
concrete sentence is the real opener, and deleting the hook improves the paragraph.
When scanning, check every opening and transition sentence for this shape explicitly.
It hides from rule-by-rule review because it violates no narrow pattern.

### 4.2 Do not manufacture a bridging sentence for a transition
> Producing that history cheaply is not enough; it must also be differentiable.

Replaced by a plain "In addition, ...". Prefer minimal connective tissue over a
rhetorical setup sentence that announces a transition is happening.

### 4.3 Do not re-illustrate what is already covered
Adding "such as the final distortion or residual stress" to a sentence about outcomes
already listed in a parenthetical a few lines above is padding. Do not add examples for
concreteness when the same ground is covered nearby.

### 4.4 Cut self-labeling, merit assertions, and superiority clauses
Three closers cut for this defect:
> that is the standard against which we ask this proposal to be judged
> This work is computational science, with its application as the proving ground
> That baseline is real

A closer is suspect when it restates the preceding sentence's mechanism as a metaphor,
self-labels the work's category or merit rather than showing it, or tells the reviewer
how to judge. Claims of category and merit belong in the sections built to argue them,
where they carry reasons. Let a paragraph end on its last concrete sentence.
The same defect appears mid-sentence as an appositive:
> makes available the object a fixed-mesh pipeline cannot form: ...

Any "the X that [competing approach] cannot do" clause is the same problem. State what
the method provides and let the contrast rest on the paragraphs that establish it.
Watch also for appeals to what an unnamed workflow, practitioner, or reviewer "needs":
"the guarantee a qualification workflow needs" is a merit assertion with a vague
beneficiary.

### 4.5 Delete a sentence that tells a contrast the neighbors show
> The two processes share almost no physics and almost no equipment.

Cut once the preceding sentences each named their own governing equations, which
exhibits the contrast directly. After writing sentences that demonstrate a difference,
check whether a neighboring sentence merely asserts it.
Caution: deleting such a sentence often strands a pronoun, because the summary sentence
is frequently where the plural noun was last named.

### 4.6 Figure cross-references are pointers, not narrations
> Figure 1 shows how the pieces compose into a single loop from recipe to outcome
>   and back
> -> Figure 1 shows the different parts of the proposed framework.

The caption carries the detail. Keep the in-text reference to a bare statement of what
the figure displays.

### 4.7 Condense a caption by diffing it against the adjacent paragraph
Captions are not required to stand alone when the figure sits beside the paragraph that
discusses it. The first move is to delete the overlap with the body text, not to hunt
for redundant wording inside the caption. This licenses deleting a claim outright
because the body carries it; it does not license compressing two claims into one
ambiguous clause.

## 5. When not to edit

These are the rules an editor most often gets wrong, and the ones worth reading twice.

### 5.1 Inelegance alone is not a defect
Minor awkwardness that does not block clarity and violates no named rule is not worth
flagging or rewriting. Reserve rewrite proposals for genuine rule violations, real
grammar errors, and unclear or ambiguous meaning, not for general "could be smoother"
judgment.
**But this is not a shield.** Before invoking it to skip a sentence, name which rules
the sentence passes. A sentence dismissed as "merely inelegant" turned out to carry a
bare demonstrative, a vague noun, and a copular relative hung off a noun-phrase list.
If any rule fails, fix it.
When genuinely in doubt, ask rather than presenting a confident rewrite.

### 5.2 Do not merge factually distinct sentences
Merging a choppy "But X. So Y." pair is right only when Y restates something already
implied by X. It is wrong when the two sentences carry distinct facts, even when those
facts are causally connected. Default to leaving causally linked but factually distinct
sentences separate; merge only genuine restatement.

### 5.3 Never blur a technical distinction to vary wording
Three sentences using the same "X is an alternative" template is a far smaller defect
than a rewrite that collapses two distinct technical actions into one verb. When a
repetition fix requires compressing two claims into one, do not do it. Either find a
variation that preserves both explicitly, or leave the repetition and flag it as
optional.

### 5.4 Do not "correct" conventional mathematical shorthand
Two rewrites made on precision grounds and deliberately reverted:
> the zero level set at time $t$ is the material deposited so far
>   ("is the boundary of the material" was rejected)
> the domain evolution is represented as one continuous function of time
>   ("the evolving domain is represented as ..." was rejected)

Both originals are standard idiom in their literature. Before proposing a precision
fix, ask whether an expert reader would actually be misled. If not, leave it.
Reserve precision flags for claims that are wrong or incomplete *as arguments*, such as
justifying a gradient from the differentiability of one factor while omitting another,
not for nouns that name a set by its interior or its boundary.

## 6. How to edit

### 6.1 Compose the replacement fully before touching the file
Rewriting one sentence four times in a row, each edit fixing the defect the previous
one introduced, is the characteristic failure. Vague demonstrative becomes pseudo-cleft
becomes too short becomes backwards causality becomes a lost domain link.
Before applying a rewrite, check the candidate against the whole rule list at once and
against the paragraph's argument: is the causal direction right, is the domain link
still explicit, is the length right. One considered edit, not a sequence of repairs.

### 6.2 Diagnose the semantic defect before touching the syntax
> developed here because the application requires it, and applicable wherever a
> computational domain evolves

This reads weak, and the instinct is to blame the parallelism. The actual defect is
that "because the application requires it" is circular: it asserts a requirement instead
of naming one. The accepted fix kept the two-postmodifier structure and replaced only
the vacuous half with the real reason.
When a phrase reads weak, first ask whether a clause carries no information. A
because-clause that restates its own main clause is the most common instance.
**Never revert instead of fixing.** Restoring the author's earlier text when they have
asked for a fix reads as abandoning the task.

### 6.3 Re-audit downstream after relocating a claim
Moving a contribution into a paragraph's opening sentence turns whatever previously
carried that claim into a restatement. A claim-first restructure always creates
redundancy candidates downstream. After moving a claim, read the whole paragraph and
delete or rewrite whatever was doing that job, salvaging the part that is still
informative.

### 6.4 Cut length by removing whole concepts, never by condensing sentences
This is the most expensive lesson in the list. Compressing prose to hit a page limit,
by merging sentences, deleting connective tissue and examples, and demoting display
equations to inline math, produced text described as "almost unreadable". Measured
damage: rewritten paragraphs at 23 to 31 words per sentence, one paragraph of 127 words
in four sentences.
Specific prohibitions:
- **Never demote a numbered display equation to inline math.** Display equations are
  white space and navigation landmarks in a dense argument, and any cross-reference to
  them degrades into an unfindable prose description.
- **Never remove lettered part markers** such as (a) through (d). They give the reader
  landing points.
- **Never delete a concrete example** that makes an abstract claim legible.
- **Never merge two factually distinct sentences** (5.2 in reverse); it always costs
  more than it saves.

When a section will not fit, identify a whole task, paragraph, or claim to delete and
propose that instead.

### 6.5 Re-read the finished paragraph against the whole list
Fixing the four instances of one rule and stopping is not a pass. A rewrite that
corrected four abstract-agent constructions simultaneously introduced a tense error, a
new abstract agent, and an improper merge. After any rewrite, read the result whole,
against every rule, as if someone else had written it.

### 6.6 Map each sentence against the argument's claims, not just the rule list
A sentence can violate no rule in this guide and still be the wrong sentence, if it
supports the wrong claim. Before editing background or motivation material, list what
each background sentence claims. When editing a downstream sentence, check which claim
it is supposed to reinforce, not only whether it is grammatically sound on its own.
> Background states two problems: manual organ segmentation, and costly manual surface
> fitting. A later sentence compares against a prior pipeline and describes it by its
> fitting-stage detail (particle-swarm initialization).
> -> Describing the same prior pipeline by its segmentation-stage detail (manual
>    annotations) ties directly back to the stated problem; the fitting-stage detail
>    does not, even though both descriptions are equally correct.

Two factually correct, individually rule-compliant sentences can differ in whether they
close the loop the background opened. Check the thread, not only the sentence.

### 6.7 A violation found once is a shape, not a single instance
Finding one abstract-technique-as-agent construction (2.4) means the shape, a named
method or technique as the grammatical subject of a verb it does not literally perform,
can recur anywhere else the same or a different technique is described the same way.
> Cross-view consensus merges the annotations ...                          (fixed)
> Differentiable NURBS refinement then fits each leaf surface ...   (same shape, found later)

After fixing one instance, search the rest of the text for the same pattern before
ending the pass. Fixing the instance you noticed and not searching for its siblings is
not a pass.

### 6.8 Editing part of a sentence obligates rereading all of it
6.3 says re-audit downstream after relocating a claim. The same discipline applies
within a single sentence: fixing one clause does not excuse skipping the rest of the
sentence you are already inside.
> A sentence's front was rewritten to fix a fronted subject (1.14). The untouched back
> half contained a parenthetical percentage stranded next to the wrong number (1.13).

Both defects were in the same sentence. Having a reason to open a sentence is a reason
to reread the whole thing, not license to touch only the clause that reason named.

### 6.9 Diagnose before repeating the last repair pattern
Applying the same successful fix a third time from pattern-matching, rather than
diagnosis, is its own failure mode. When a fronted or stranded phrase is the defect,
check first whether the phrase is worth keeping at all (4.1-4.7) before reaching for
the standard relocate-and-bracket repair (1.14).
> ..., evaluated against the curated organ labels, recovered $1{,}017$ of $1{,}023$
> reference leaves ...                                              (relocated, redundant)
> recovered $1{,}017$ of $1{,}023$ reference leaves ...              (deleted, correct)

The fronted phrase was redundant with "reference leaves" stated a few words later.
Deleting it was the fix; relocating it to a bracketed modifier only repeated a template
that fit the previous two sentences, not this one.

## 7. Mechanical checks

Some rules reduce to a search. Run these over the whole document, not the paragraph
under discussion; violations predating the current pass survive style reviews
indefinitely.

```bash
grep -n "—" FILE                                    # em-dashes
grep -nE "(is|are) what [a-z]+" FILE                # pseudo-cleft, forward
grep -nE "^What .+ (is|are) " FILE                  # pseudo-cleft, inverted
grep -nE "\b(requires?|needs?) no\b" FILE           # "requires no X"
grep -nE "\b(affordable|cheaply?|expensive|pays for itself)\b" FILE
grep -nE "\b(consumed?|inherits?|drives?|feeds?|supplies|absorbs?|harness)\b" FILE
grep -nE "\b(machinery|settings|factors|aspects|things)\b" FILE
grep -nE "^(It|This|That|These|Those) [a-z]+" FILE  # bare demonstrative openers
grep -nE "\b(directly|exactly|precisely|fully|simply|readily|naturally)\b" FILE
grep -nE "of [a-z]+ of [a-z]+ of " FILE             # stacked "of" chains
grep -nE "; [a-z]" FILE                             # semicolon fragments, then read each
grep -nE "\b(focusses|modelling|behaviour|analyse|centre)\b" FILE   # UK/US drift
```

**A regex sweep is not a pass.** A search for abstract agents that requires the subject
to be followed immediately by its verb misses "this proposal *already* uses" and "the
rest of the objective uses", because an adverb or an embedded "of" phrase intervenes.
Three consecutive sweeps reported clean and the author found more instances each time.
Treat grep as a way to find the easy cases before reading, never as the reading itself.

Reduced relative clauses (1.1), abstract agents (2.4), misattached relative clauses
(1.9), and elided verbs (1.11) do not reduce to a regex at all. They require reading
every sentence whose subject is an abstract noun, and reading each "which" clause
against the noun immediately before it.

The semicolon and adverb greps above are noisy by design: most hits are legitimate.
They are worth running because the illegitimate ones are invisible on a read-through.

## 8. One-page checklist

Before proposing any rewrite:

1. Does the sentence contain a reduced relative, a pseudo-cleft, or "requires no X"?
2. Is the real subject fronted behind a participial, absolute, or gerund-phrase
   modifier? If so, put the subject first.
3. Is every clause independent where the punctuation says it should be?
4. Does each relative clause attach to the noun it means, and does no clause borrow a
   verb from across an "and"?
5. Are there three "of" phrases in a row, or a prepositional phrase stranded from what
   it modifies?
6. Is any verb figurative, weak, colloquial, subfield jargon, or economic?
7. Is an abstract object the agent of an action only a person or program performs?
8. Does any adverb survive being deleted? If so, delete it.
9. Is any noun a placeholder, or any demonstrative pointing at something unnamed?
10. Ask "What is that?" of every opener and every closer.
11. Does the sentence self-label merit, category, or superiority?
12. Does it tell a contrast the neighboring sentences already show?
13. Is the tense right for work not yet done?
14. Is there notation in a section written for a different reader?
15. Would the fix blur a technical distinction, or correct conventional shorthand?
    If so, stop.
16. If this is background, motivation, or a connective sentence, which claim in the
    surrounding argument is it supposed to reinforce? An equally correct alternative
    detail that does not reinforce that claim is the wrong choice (6.6).
17. If the defect you are about to fix is the same shape as one you already fixed in
    this pass, search the rest of the document for other instances before fixing only
    this one (6.7).
18. If the fix is the same repair pattern you used for the previous one or two
    instances, check first whether the triggering content is worth keeping at all; a
    deletion may be simpler than another relocation (6.9).

After applying it:

19. Read the whole paragraph against all eighteen again. Fixing four instances of one
    rule and stopping is not a pass; the rewrite itself is new text.
20. Reread the entire sentence you edited, not only the clause you changed. A defect
    can be sitting in the untouched half (6.8).
21. Check that nothing upstream or downstream now restates a moved claim.
22. Confirm no antecedent was stranded by a deletion.
