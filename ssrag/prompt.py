"""Prompts for SS-RAG — entity / correlation extraction & QA (AR + EN)."""
ENTITY_EXTRACT_PROMPT = """You are an expert knowledge extractor for a hypergraph RAG system.

Given a TEXT CHUNK, extract:
1. ENTITIES: distinct concepts, people, places, diseases, drugs, methods, events. For each give: name, type, one-sentence description in the chunk's language.
2. LOW-ORDER correlations (pairwise, exactly 2 entities): (entity_A, entity_B, keywords, one-sentence description).
3. HIGH-ORDER correlations (hyperedges, 3+ entities that jointly form one fact/mechanism/event): (entity list, keywords, one-sentence description).

Rules:
- Names must appear verbatim (or close paraphrase) in the chunk.
- Prefer 3-8 entities per chunk. Skip generic stopwords.
- Output STRICT JSON only, no markdown, matching this schema:
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}],
  "low_order": [{{"entities": ["A", "B"], "keywords": "k1, k2", "description": "..."}}],
  "high_order": [{{"entities": ["A", "B", "C"], "keywords": "k1, k2", "description": "..."}}]}}

TEXT CHUNK:
{chunk}
"""

QA_SYSTEM_PROMPT = """You are SS-RAG, a careful assistant. Answer ONLY from the provided CONTEXT.
If the context is insufficient, say so explicitly (in the user's language) instead of hallucinating.
Cite entity/relation evidence briefly when possible.
Response type: {response_type}.
User language: follow the QUESTION language (Arabic or English)."""

QA_USER_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}

Answer concisely but completely."""
