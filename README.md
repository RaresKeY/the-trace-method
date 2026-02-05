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

## Example of actual flow: NLP Training

### 1. Initial Ask (Prompt 1)
Read `@Training an NLP model from scratch.md` and find out how far I am from the actual process used in `@train.py`. Refactor the file for better formatting while keeping the style of the speaker. Do not hallucinate data. Correct pacing, concepts, and mentions, organizing it logically for a learner. First, clearly mark corrected concepts in a markdown file by quoting what is wrong and providing the correction.

### 2. Technical Details - Ask what you don't understand (Prompt 2)
Add a technical details section (linked in preamble). Explain concepts conceptually and link key words to the technical section containing concise examples and explanations:
- How do we calculate loss? 
- How do we vectorize (manual vs. random)? 
- How do we nudge neurons and connections during training? 
- What are the detailed steps in the training process?
- Advanced section: How to use ngrams, fasttext, and other modern techniques.

### 3. Refine Style (Prompt 3)
Add refined conceptual explanations and examples that align with the style of the original text.

### 4. Final Touch (Prompt 4)
Format `@Training an NLP model from scratch.md` for final delivery.