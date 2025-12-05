"""
kerala_rag_demo.py
Simple RAG prototype for the Kerala Ayurveda content pack.
Save and run: python kerala_rag_demo.py
"""

import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# ======================================================================
# Data classes
# ======================================================================

@dataclass
class Citation:
    doc_id: str
    section_id: str
    snippet: str = ""

@dataclass
class Document:
    doc_id: str
    section_id: str
    content: str
    metadata: Dict[str, Any]

# ======================================================================
# Simple RAG system
# ======================================================================

class SimpleRAGSystem:
    """
    Lightweight RAG for the provided Kerala Ayurveda corpus.
    Hybrid retrieval: BM25-like TF scoring + metadata boosts + phrase bonus.
    """

    def __init__(self):
        self.documents: List[Document] = []
        self.corpus_loaded = False

    # -------------------------
    # Corpus loading & chunking
    # -------------------------
    def load_corpus(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            doc_type = doc.get('type', 'text')
            doc_id = doc['id']
            if doc_type == 'markdown':
                chunks = self._chunk_markdown(doc_id, doc['content'], doc.get('metadata', {}))
            elif doc_type == 'faq':
                chunks = self._chunk_faq(doc_id, doc['content'], doc.get('metadata', {}))
            elif doc_type == 'csv':
                chunks = self._chunk_csv(doc_id, doc['content'], doc.get('metadata', {}))
            else:
                chunks = [Document(doc_id, 'main', doc['content'], doc.get('metadata', {}))]
            self.documents.extend(chunks)
        self.corpus_loaded = True
        print(f"[INFO] Loaded {len(self.documents)} document chunks")

    def _slugify(self, text: str) -> str:
        s = text.lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        s = re.sub(r'_{2,}', '_', s)
        return s.strip('_')

    def _chunk_markdown(self, doc_id: str, content: str, metadata: Dict) -> List[Document]:
        chunks = []
        sections = re.split(r'\r?\n##\s+', content)
        if sections:
            intro = sections[0].strip()
            if intro:
                chunks.append(Document(doc_id=doc_id, section_id='intro', content=intro, metadata=metadata))
        for i, section in enumerate(sections[1:], 1):
            lines = section.split('\n', 1)
            section_title = lines[0].strip()
            section_content = lines[1].strip() if len(lines) > 1 else ''
            full_content = f"## {section_title}\n\n{section_content}"
            chunks.append(Document(
                doc_id=doc_id,
                section_id=f"section_{i}_{self._slugify(section_title)}",
                content=full_content,
                metadata={**metadata, 'section_title': section_title}
            ))
        return chunks

    def _chunk_faq(self, doc_id: str, content: str, metadata: Dict) -> List[Document]:
        chunks = []
        # Match '## <num>. Question' followed by answer until next '## <num>.' or EOF
        pattern = re.compile(r'##\s*\d+\.\s*(.+?)\n\n(.*?)(?=\n##\s*\d+\.|\Z)', re.S)
        matches = pattern.findall(content)
        if not matches:
            # fallback: treat whole faq as single chunk
            return [Document(doc_id=doc_id, section_id='faq_all', content=content, metadata=metadata)]
        for i, (question, answer) in enumerate(matches, 1):
            full_content = f"Q: {question.strip()}\n\nA: {answer.strip()}"
            chunks.append(Document(
                doc_id=doc_id,
                section_id=f"faq_{i}_{self._slugify(question)}",
                content=full_content,
                metadata={**metadata, 'faq_number': i, 'question': question.strip()}
            ))
        return chunks

    def _chunk_csv(self, doc_id: str, content: List[Dict], metadata: Dict) -> List[Document]:
        chunks = []
        for row in content:
            formatted = self._format_product_row(row)
            prod_id = row.get('product_id') or str(hash(row.get('name', '')))
            chunks.append(Document(
                doc_id=doc_id,
                section_id=f"product_{prod_id}",
                content=formatted,
                metadata={**metadata, **row}
            ))
        return chunks

    def _format_product_row(self, row: Dict) -> str:
        parts = [
            f"Product: {row.get('name', 'Unknown')}",
            f"ID: {row.get('product_id', 'N/A')}",
            f"Category: {row.get('category', 'N/A')}",
            f"Format: {row.get('format', 'N/A')}",
            f"Target concerns: {row.get('target_concerns', 'N/A')}",
            f"Key herbs: {row.get('key_herbs', 'N/A')}",
            f"Contraindications: {row.get('contraindications_short', 'N/A')}",
            f"Tags: {row.get('internal_tags', 'N/A')}"
        ]
        return "\n".join(parts)

    # -------------------------
    # Retrieval
    # -------------------------
    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        if not self.corpus_loaded:
            return []
        query_terms = self._tokenize(query.lower())
        scored_docs: List[Tuple[Document, float]] = []
        for doc in self.documents:
            score = self._score_document(doc, query_terms, query.lower())
            if score > 0:
                scored_docs.append((doc, score))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _score_document(self, doc: Document, query_terms: List[str], query_raw: str) -> float:
        doc_text = doc.content.lower()
        doc_tokens = self._tokenize(doc_text)
        if not doc_tokens:
            return 0.0
        score = 0.0
        # TF-like scoring
        for term in query_terms:
            tf = doc_tokens.count(term)
            if tf > 0:
                score += (tf / len(doc_tokens)) * 10.0
        # Coverage
        matching_terms = sum(1 for term in query_terms if term in doc_tokens)
        coverage = matching_terms / len(query_terms) if query_terms else 0.0
        score += coverage * 5.0
        # Exact phrase bonus
        if query_raw in doc_text:
            score += 15.0
        # Metadata boost
        metadata_text = ' '.join(str(v) for v in doc.metadata.values()).lower()
        for term in query_terms:
            if term in metadata_text:
                score += 3.0
        return score

    # -------------------------
    # Answering
    # -------------------------
    def answer_user_query(self, query: str) -> Dict[str, Any]:
        retrieved = self.retrieve(query, top_k=4)
        if not retrieved:
            return {
                'answer': "No relevant information found in the Kerala Ayurveda corpus.",
                'citations': [],
                'confidence': 'low'
            }
        context_parts = []
        citations: List[Citation] = []
        for i, (doc, score) in enumerate(retrieved, 1):
            header = f"[Source {i}] {doc.doc_id} | {doc.section_id}"
            # Truncate content for context safety in a demo
            snippet = doc.content if len(doc.content) <= 1500 else doc.content[:1500] + "..."
            context_parts.append(f"{header}\n{snippet}\n")
            citations.append(Citation(doc_id=doc.doc_id, section_id=doc.section_id, snippet=snippet[:120]))
        context = "\n".join(context_parts)

        # Simulated generation (replace with LLM call in production).
        answer_text = self._generate_answer(query, context, retrieved)

        return {
            'answer': answer_text,
            'citations': [{'doc_id': c.doc_id, 'section_id': c.section_id, 'snippet': c.snippet} for c in citations],
            'confidence': 'high' if len(retrieved) >= 2 else 'medium'
        }

    def _generate_answer(self, query: str, context: str, retrieved: List[Tuple[Document, float]]) -> str:
        # Production: call LLM with a prompt like:
        # "Only use the following sources. Answer concisely and attach inline numeric citations (1),(2)..."
        # Demo: extract first meaningful sentences from top 2 retrieved
        parts = []
        for doc, _ in retrieved[:2]:
            sentences = re.split(r'(?<=[.!?])\s+', doc.content.strip())
            meaningful = [s.strip() for s in sentences if len(s.strip()) > 40]
            if meaningful:
                parts.append(meaningful[0])
        if parts:
            # Append mapping hint to allow caller to map (1),(2) -> doc ids
            return " ".join(parts)
        # Fallback: return first 200 chars of top doc
        topdoc = retrieved[0][0].content
        return topdoc[:400] + ("..." if len(topdoc) > 400 else "")

# ======================================================================
# Example corpus & test runner
# ======================================================================

def create_sample_corpus() -> List[Dict[str, Any]]:
    corpus = [
        {
            'id': 'ayurveda_foundations',
            'type': 'markdown',
            'content': '''# Ayurveda Foundations – Internal Primer

## What is Ayurveda?

Ayurveda is a traditional system of health that originated in India. It focuses on understanding an individual's constitution (prakriti), maintaining balance between body, mind, and environment, and using diet, lifestyle, herbs, and therapies to support wellbeing.

## The Tridosha Model (Vata, Pitta, Kapha)

Ayurveda groups functions of the body–mind into three broad principles called doshas: Vata (movement, communication), Pitta (transformation, digestion, metabolism), and Kapha (stability, structure, nourishment).''',
            'metadata': {'category': 'foundations', 'type': 'educational'}
        },
        {
            'id': 'faq_general',
            'type': 'faq',
            'content': '''# FAQ – General Ayurveda Questions

## 1. Is Ayurveda safe to combine with modern medicine?

Ayurveda is often used alongside modern medicine. However, herb–drug interactions are possible. We encourage readers to inform their doctor about any Ayurvedic supplements or therapies they use.

## 2. How long does it take to see results?

Timelines vary from person to person. Some people may feel changes in sleep, digestion, or energy in a few weeks. Deeper changes often take longer. We recommend thinking in terms of weeks to months, not overnight fixes.

## 3. Can Ayurveda help with stress and sleep?

Ayurveda approaches stress and sleep through daily routines, food choices, herbs like Ashwagandha, and therapies such as oil massages. Ayurvedic support can complement, but not replace, professional mental health care when needed.''',
            'metadata': {'category': 'faq', 'type': 'customer_support'}
        },
        {
            'id': 'product_catalog',
            'type': 'csv',
            'content': [
                {
                    'product_id': 'KA-P001',
                    'name': 'Triphala Capsules',
                    'category': 'Digestive support',
                    'format': 'Capsules',
                    'target_concerns': 'Digestive comfort; regular elimination',
                    'key_herbs': 'Amalaki; Bibhitaki; Haritaki',
                    'contraindications_short': 'Consult doctor in chronic digestive disease, pregnancy, or post-surgery',
                    'internal_tags': 'digestion; gut health; elimination'
                },
                {
                    'product_id': 'KA-P002',
                    'name': 'Ashwagandha Stress Balance Tablets',
                    'category': 'Stress & Sleep',
                    'format': 'Tablets',
                    'target_concerns': 'Stress resilience; restful sleep',
                    'key_herbs': 'Ashwagandha root extract',
                    'contraindications_short': 'Caution in thyroid/autoimmune conditions, pregnancy',
                    'internal_tags': 'stress; sleep; adaptogen; nervous system'
                },
                {
                    'product_id': 'KA-P003',
                    'name': 'Brahmi Tailam – Head & Hair Oil',
                    'category': 'Topical oil',
                    'format': 'Oil',
                    'target_concerns': 'Scalp nourishment; relaxation',
                    'key_herbs': 'Brahmi; Amla',
                    'contraindications_short': 'External use only; patch-test for sensitive skin',
                    'internal_tags': 'hair & scalp; vata soothing; evening ritual'
                }
            ],
            'metadata': {'category': 'products', 'type': 'catalog'}
        },
        {
            'id': 'stress_support_program',
            'type': 'markdown',
            'content': '''# Clinic Program – Stress Support Protocol

## Overview

Name: Stress Support Program
Setting: Kerala Ayurveda clinics
Focus: Supporting people who experience ongoing stress, mental fatigue, and difficulty unwinding.

## Core Components

1. Initial Ayurvedic Consultation - Detailed history of lifestyle, sleep, digestion, stressors
2. Therapy Plan - Abhyanga (oil massage), Shirodhara (oil stream over forehead)
3. Home Routine Suggestions - Evening wind-down ideas, regular mealtimes

## Safety Note

This program is not a substitute for mental health care. If you experience severe mood changes or thoughts of self-harm, please seek support from a qualified mental health professional.''',
            'metadata': {'category': 'programs', 'type': 'clinic_services'}
        }
    ]
    return corpus

def run_example_queries():
    rag = SimpleRAGSystem()
    corpus = create_sample_corpus()
    rag.load_corpus(corpus)

    queries = [
        "What are the key benefits of Ashwagandha Stress Balance Tablets?",
        "Are there any contraindications for Triphala Capsules?",
        "Can Ayurveda help with stress and sleep?"
    ]

    for i, query in enumerate(queries, 1):
        print("\n" + "="*80)
        print(f"QUERY {i}: {query}")
        print("="*80)
        result = rag.answer_user_query(query)
        print("\nANSWER:\n", result['answer'])
        print("\nCITATIONS:")
        for j, c in enumerate(result['citations'], 1):
            print(f"  [{j}] {c['doc_id']} | {c['section_id']} -> {c['snippet'][:100]}...")
        print("\nCONFIDENCE:", result['confidence'])
        print("\nPOTENTIAL FAILURE MODES:")
        if 'contraindication' in query.lower():
            print("  - May miss contraindications found in other docs; merging notes incorrectly.")
        elif 'benefit' in query.lower():
            print("  - Risk of paraphrasing to imply stronger efficacy than corpus language.")
        else:
            print("  - Might produce generalised advice lacking user-specific nuance.")

if __name__ == "__main__":
    run_example_queries()
