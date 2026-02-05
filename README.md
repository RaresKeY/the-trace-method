# LogicLoop 🌀

> From "It just works" to "I know why it works."

LogicLoop is a systematic framework for mastering technical concepts by reverse-engineering functional code through AI-assisted auditing.

## The Workflow
1. **The Black Box:** Start with a working script.
2. **The Synthesis:** Write your mental model of how it works.
3. **The Audit:** Use AI to cross-reference your model against the code.
4. **The Refinement:** Produce a high-fidelity "White Box" document.

---

## The Learning Loop in Practice 
*Pro Tip: Rewriting your draft from scratch for each loop forces active recall; this is where the real learning happens.*

- **Build:** Create a working 'black box' script. *(Secure a quick win and a dopamine hit to fuel the rest of the loop.)*
- **Explore:** Discuss the logic with AI. *(Stress-test your assumptions and explore the mechanism playfully.)*
- **Synthesize:** Draft the concepts as if teaching a peer. *(The Feynman Technique: Imagine you are explaining the cool new thing you just discovered.)*
- **Audit:** Use AI to generate a correction log against the code. *(Ground fluid ideas into known truths. Skip the "50-book detour" and replace floating assumptions with concrete conviction—giving your brain a solid structural block to build upon, discarding the "stuck/unsure" state, and freeing your attention.)*
- **Polish:** Let the AI structure and organize the final document. *(Transform rough notes into a high-fidelity reference you'll actually return to.)*

## Case Study: The NLP Loop
*A real-world example of using LogicLoop to master a simple Neural Network.*

### Phase 1: The Gap Analysis
**Goal:** Identify exactly where my "Mental Model" drifts from the code's reality.
**The Prompt:**
> "Read `@draft.md` and find out how far I am from the actual process used in `@train.py`. Refactor the file for better formatting while keeping my style. **First, clearly mark corrected concepts in a markdown file by quoting what is wrong and providing the correction.**"

### Phase 2: The Deep Dive
**Goal:** Fill the specific knowledge gaps identified in Phase 1.
**The Prompt:**
> "Add a 'Technical Details' section. Explain these concepts conceptually and link key words to concise technical examples:
> - How do we calculate loss?
> - How do we vectorize (manual vs. random)?
> - How do we nudge neurons/connections?
> - Advanced: How to use ngrams/fasttext?"

### Phase 3: Style Alignment
**Goal:** Ensure the new technical knowledge blends seamlessly with my original voice.
**The Prompt:**
> "Add refined conceptual explanations and examples that align with the style of the original text.

### Phase 4: Final Polish
**Goal:** Structure the document for long-term storage.
**The Prompt:**
> "Format `@final.md` for delivery. Clean up the headers, ensure links work, and make it pleasant to read."