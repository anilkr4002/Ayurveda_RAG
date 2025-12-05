


# Kerala Ayurveda – RAG System Assignment 

# Part A – Small RAG Design 

## 1. High-Level RAG Approach 

1. **How we split the documents (Chunking):**

   * Markdown files are divided according to `##` headings. The introduction becomes its own chunk.
   * FAQs are split so each question–answer pair becomes one chunk.
   * The product CSV is split so each product becomes one chunk, written in normal readable text.
   * Every chunk includes helpful context like section titles.
   * No sliding window is needed because the dataset is small and clear.

2. **How we retrieve information:**

   * We use a **hybrid retrieval** method that mixes BM25-style keyword matching with metadata‑based boosts.
   * Because the dataset is small, adding embeddings doesn’t give major benefit.

3. **How many pieces we retrieve (Top‑K):**

   * We pull the **top 4 chunks**, but only give the **top 3** to the LLM.
   * If less than 2 chunks are relevant, we mark the answer as lower‑confidence.

4. **How we give context to the LLM:**

   * Each chunk is formatted like this: `[Source N] <doc_id> | <section_id>` followed by the chunk text.
   * Long chunks are shortened to around 800–1500 characters.

5. **Citation format:**

   * Every chunk carries `{doc_id, section_id}` metadata.
   * The LLM shows citations as simple numbers like (1), (2), etc.

6. **Safety rules:**

   * The model **must answer using only the given sources**.
   * If the answer is not in the corpus, it must say so.

---

## 2. RAG Function Design (Pseudo-Code)

```
answer_user_query(query):

    # Step 1: Retrieve relevant chunks
    retrieved = retriever.get_top_k(query, k=4)
    if nothing found:
        return "No relevant info found"

    # Step 2: Build the LLM context
    context = ""
    citations = []

    for each chunk:
        context += "[Source X] metadata + content"
        add its citation info to citations

    # Step 3: Create the final prompt
    prompt = "Answer only using these sources. Use citations."

    # Step 4: Generate answer using LLM
    answer = llm.generate(prompt)

    # Step 5: Send answer + citations back
    return { answer, citations }
```

---

## 3. Example Queries

### Example 1 — *Benefits of Ashwagandha Stress Balance Tablets*

**Expected retrieved sections:**

* `product_catalog` → `product_KA-P002`
* `faq_general` → FAQ #3 (stress & sleep)
* `stress_support_program` → Overview or Core Components (context on stress support)

**Sample answer:**

> These tablets support the body’s stress response and help promote better sleep (1). Ashwagandha traditionally helps maintain emotional balance, and the FAQ highlights its role in sleep support (2). These benefits work alongside lifestyle practices suggested in the stress support program (3).

**Possible issue:** The model might accidentally make medical claims beyond the corpus.

---

### Example 2 — *Contraindications for Triphala Capsules*

**Expected retrieved sections:**

* `product_catalog` → `product_KA-P001`
* `faq_general` → FAQ #1 (safety with medicines)

**Sample answer:**

> Triphala Capsules should be used with caution in conditions like chronic digestive issues, pregnancy, or after surgery (1). The FAQ also reminds that herbs may interact with medicines and should be taken with professional guidance (2).

**Possible issue:** Some safety notes may be missed if not chunked correctly.

---

### Example 3 — *Can Ayurveda help with stress and sleep?*

**Expected retrieved sections:**

* `faq_general` → FAQ #3
* `stress_support_program` → Overview + Core Components

**Sample answer:**

> Ayurveda supports stress relief and sleep through calming routines, grounding foods, herbs such as Ashwagandha, and therapies like oil massage and Shirodhara (1). The Stress Support Program also emphasizes guided routines and relaxation habits (2).

**Possible issue:** The model must avoid giving personalised medical advice.

---

# Part B – Agent Workflow & Evaluation

## 1. Agent Workflow (4 Simple Steps)

### **Step 1 – Brief Intake Agent**

* Converts the user’s request into a structured plan.
* Makes sure at least one valid source is referenced.


* **Role:** Convert user brief into structured content spec.
* **Input:** `{brief, audience, length, tone}`
* **Output:** `{title, outline_constraints, allowed_sources, safety_requirements}`
* **Failure mode:** Misunderstanding scope.
* **Guardrail:** Require at least 1 source tag from corpus before proceeding.

### **Step 2 – Outline Agent**

* Builds an outline fully grounded in the corpus.
* Every heading must link to at least one real source.

* **Role:** Produce an outline grounded in available sections.
* **Input:** `{title, allowed_sources}`
* **Output:** `{outline: [{heading, source_links}], est_wordcount}`
* **Failure mode:** Outline requires unsupported claims.
* **Guardrail:** Each heading must map to at least one corpus section.

### **Step 3 – Writer Agent**

* Writes the article based strictly on the approved sources.
* Every paragraph must contain a citation.

* **Role:** Write article using only approved sources.
* **Input:** `{outline, sources, tone_rules}`
* **Output:** `{draft, inline_citations}`
* **Failure mode:** Hallucination.
* **Guardrail:** Reject any paragraph without at least one grounded source.

### **Step 4 – Fact‑Checker & Tone Editor**

* Checks facts, ensures tone consistency, and flags unsafe statements.

* **Role:** Validate facts and apply brand tone.
* **Input:** `{draft, inline_citations}`
* **Output:** `{checked_draft, flags, citations}`
* **Failure mode:** Missing warnings or tone mismatches.
* **Guardrail:** Automatic flag for unsupported statements or medical‑boundary violations.


---

## 2. Evaluation Loop 

### Golden Set Includes:

* Product KA‑P002 description
* FAQ on Triphala precautions
* Stress Support Program intro
* Negative tests like medical claims or dosage requests

### We Score On:

* Proper citations
* Hallucination rate
* Tone correctness
* Amount of editor corrections needed

### Metrics To Track:

* % of drafts accepted without major edits
* Count of safety issues per draft
* Avg factual errors per draft

---

## 3. Prioritisation for First 2 Weeks

### Ship Now (MVP):

* RAG‑based internal Q&A with citations
* End‑to‑end pipeline: Brief → Outline → Draft → Check
* Logging + evaluation using golden set

### Postpone:

* Advanced editor UI
* Multi‑language features
* External evidence integration

---

# Short Reflection 

* Time spent: around 90 minutes.
* Most interesting part: designing guardrails that maintain accuracy and brand tone.
* Still unclear: how closely wording must match the original team’s style.
* AI usage: helped structure ideas, rewrite, and check clarity—while keeping strictly to the provided content.
