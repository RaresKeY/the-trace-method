# LogicLoop 🌀

> From "It just works" to "I know why it works."

LogicLoop is a systematic framework for mastering technical concepts by reverse-engineering functional code through AI-assisted auditing.

## The Workflow
1. **The Black Box:** Start with a working script.
2. **The Synthesis:** Write your mental model of how it works.
3. **The Audit:** Use AI to cross-reference your model against the code.
4. **The Refinement:** Produce a high-fidelity "White Box" document.

---

## Learning Loop used in practice 
*Note: Works better if you discard and rewrite your original md from scratch on new loop, it solidifies newly learned concepts in your mind*

- Make a working 'black box' script.
- Discuss with AI how things work.
- Lay out concepts as if teaching someone.
- Use AI to generate a corrections MD aligned with script functionality.
- Let the AI format, clean, and organize the original document. (makes reading it more pleasant)

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