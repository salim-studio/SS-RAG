"""Core operations: extraction, hypergraph build, retrieval + generation.

Mirrors Hyper-RAG's pipeline but simplified:
  insert = chunk -> extract entities / low-order / high-order (LLM or offline regex)
           -> upsert hypergraph + vector DBs
  query  = retrieve seed entities/chunks -> 1-hop hypergraph diffusion
           -> build context -> LLM answer
Modes: hyper (full), hyper-lite (entities+chunks only), graph (pairwise only),
       naive (chunks only), llm (no retrieval).
"""
from __future__ import annotations
import json
import re
from typing import Dict, List

from .prompt import ENTITY_EXTRACT_PROMPT, QA_SYSTEM_PROMPT, QA_USER_TEMPLATE
from .utils import (
    chunking_by_token_size, clean_str, truncate_list_by_token_size,
    num_tokens, logger,
)


def _parse_llm_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "this", "that", "it", "as", "at", "by", "from",
    "هذا", "هذه", "التي", "الذي", "على", "في", "من", "إلى", "أن", "إن", "كان",
    "ما", "مع", "بين", "عن", "تم", "قد", "كل", "بعد", "قبل", "عند", "غير",
}


def offline_extract(chunk_text: str) -> dict:
    """Regex fallback: entities = frequent content words (EN+AR), hyperedges = sentence co-occurrence."""
    words = re.findall(r"[\u0600-\u06FF\w\-]{3,}", chunk_text)
    freq: Dict[str, int] = {}
    for w in words:
        lw = w.strip(".,;:!?()\"'«»")
        if not lw or lw.lower() in _STOP:
            continue
        freq[lw] = freq.get(lw, 0) + 1
    entities = sorted(freq, key=lambda k: -freq[k])[:8]
    ent_objs = [{"name": e, "type": "concept", "description": f"Mentioned in chunk ({freq[e]}x)."} for e in entities]

    sentences = re.split(r"(?<=[.!?؟\n])\s+", chunk_text)
    low, high, seen = [], [], set()
    for s in sentences:
        present = [e for e in entities if e in s]
        if len(present) == 2:
            key = tuple(sorted(present))
            if key not in seen:
                seen.add(key)
                low.append({"entities": list(key), "keywords": ", ".join(key),
                            "description": s.strip()[:220]})
        elif len(present) >= 3:
            key = tuple(sorted(present))
            if key not in seen:
                seen.add(key)
                high.append({"entities": list(key), "keywords": ", ".join(key[:4]),
                             "description": s.strip()[:260]})
    return {"entities": ent_objs, "low_order": low, "high_order": high}


async def extract_entities(chunks: dict, hypergraph_inst, entity_vdb, relationships_vdb, global_config: dict):
    llm_func = global_config.get("llm_model_func")
    use_llm = global_config.get("use_llm_extract", True) and llm_func is not None
    # quick offline check: llm module returns offline echo when no key; detect key presence
    try:
        from .llm import llm_configured
        if not llm_configured():
            use_llm = False
    except Exception:
        pass

    for chunk_key, chunk in chunks.items():
        text = chunk.get("content", "")
        data = None
        if use_llm:
            try:
                prompt = ENTITY_EXTRACT_PROMPT.format(chunk=text[:6000])
                resp = await llm_func(prompt)
                data = _parse_llm_json(resp or "")
            except Exception as e:
                logger.warning(f"LLM extract failed for {chunk_key}: {e}")
                data = None
        if data is None:
            data = offline_extract(text)

        # upsert vertices
        for ent in data.get("entities", []):
            name = clean_str(ent.get("name", "")).strip()
            if not name:
                continue
            await hypergraph_inst.upsert_vertex(name, {
                "entity_name": name,
                "entity_type": ent.get("type", "concept"),
                "description": ent.get("description", "")[:500],
                "source_ids": [chunk_key],
            })
        # upsert hyperedges (low + high)
        for rel in data.get("low_order", []):
            nodes = [clean_str(x).strip() for x in rel.get("entities", []) if x]
            if len(nodes) != 2:
                continue
            eid = hypergraph_inst._eid(nodes, "low")
            await hypergraph_inst.upsert_hyperedge(eid, nodes, {
                "order": "low", "keywords": rel.get("keywords", ""),
                "description": rel.get("description", "")[:600],
                "source_ids": [chunk_key],
            })
        for rel in data.get("high_order", []):
            nodes = [clean_str(x).strip() for x in rel.get("entities", []) if x]
            if len(nodes) < 2:
                continue
            order = "high" if len(nodes) >= 3 else "low"
            eid = hypergraph_inst._eid(nodes, order)
            await hypergraph_inst.upsert_hyperedge(eid, nodes, {
                "order": order, "keywords": rel.get("keywords", ""),
                "description": rel.get("description", "")[:600],
                "source_ids": [chunk_key],
            })

    # (re)build vector DBs from hypergraph — simple & robust
    verts = await hypergraph_inst.get_all_vertices()
    if verts:
        await entity_vdb.upsert({
            f"ent-{k}": {"content": f"{v.get('entity_name', k)}: {v.get('description', '')}"}
            for k, v in verts.items()
        })
    edges = await hypergraph_inst.get_all_hyperedges()
    if edges:
        await relationships_vdb.upsert({
            eid: {"content": f"{e.get('keywords', '')}: {e.get('description', '')} | nodes={','.join(e.get('nodes', []))}"}
            for eid, e in edges.items()
        })
    return hypergraph_inst


# ---------------- retrieval helpers ----------------

async def _seed_entities(query: str, entity_vdb, top_k: int) -> List[dict]:
    try:
        return await entity_vdb.query(query, top_k=top_k)
    except Exception:
        return []


async def _diffuse(seed_names: List[str], hypergraph_inst, include_high: bool = True) -> List[dict]:
    """1-hop diffusion: seed entities -> hyperedges -> neighbour entities."""
    out, seen_e = [], set()
    for name in seed_names:
        # match vertex id either exactly or via ent- prefix
        vid = name
        if not await hypergraph_inst.has_vertex(vid):
            # try lookup by entity_name
            verts = await hypergraph_inst.get_all_vertices()
            match = next((k for k, v in verts.items() if v.get("entity_name") == name), None)
            if match is None:
                continue
            vid = match
        for eid in await hypergraph_inst.get_nbr_e_of_vertex(vid):
            if eid in seen_e:
                continue
            e = await hypergraph_inst.get_hyperedge(eid)
            if not e:
                continue
            if not include_high and e.get("order") == "high":
                continue
            seen_e.add(eid)
            out.append({"id": eid, **e})
    return out


def _entity_name_of(hit: dict) -> str:
    c = hit.get("content", "")
    return c.split(":")[0].strip()[:80] or hit.get("id", "")


def _build_context(entity_hits, rels, chunk_hits, text_chunks_store, param) -> str:
    ent_lines = []
    for h in truncate_list_by_token_size(entity_hits, key=lambda x: x.get("content", ""), max_token_size=param.max_token_for_entity_context):
        ent_lines.append(f"- {h.get('content', '')[:300]}")
    rel_lines = []
    for r in truncate_list_by_token_size(rels, key=lambda x: x.get("description", "") if isinstance(x, dict) else "", max_token_size=param.max_token_for_relation_context):
        nodes = ",".join(r.get("nodes", []))
        rel_lines.append(f"- [{r.get('order', '?')}] ({nodes}): {r.get('description', '')[:300]}")
    chunk_lines = []
    for h in truncate_list_by_token_size(chunk_hits, key=lambda x: x.get("content", ""), max_token_size=param.max_token_for_text_unit):
        chunk_lines.append(f"--- chunk {h.get('id', '')} ---\n{h.get('content', '')[:1500]}")
    return (
        "## Entities\n" + ("\n".join(ent_lines) or "(none)") + "\n\n"
        "## Correlations (hyperedges)\n" + ("\n".join(rel_lines) or "(none)") + "\n\n"
        "## Source chunks\n" + ("\n".join(chunk_lines) or "(none)")
    )


async def _generate(query: str, context: str, param, global_config, only_context=False):
    if only_context or param.only_need_context:
        return context
    llm_func = global_config["llm_model_func"]
    sys_p = QA_SYSTEM_PROMPT.format(response_type=param.response_type)
    user_p = QA_USER_TEMPLATE.format(context=context[:12000], question=query)
    try:
        return await llm_func(user_p, system_prompt=sys_p)
    except Exception as e:
        return f"[SS-RAG generation error: {e}]\n\n{context[:2000]}"


# ---------------- public query modes ----------------

async def hyper_query(query, hg, ent_vdb, rel_vdb, text_chunks, param, global_config):
    ent_hits = await _seed_entities(query, ent_vdb, param.top_k)
    seeds = [_entity_name_of(h) for h in ent_hits]
    rels = await _diffuse(seeds, hg, include_high=True)
    chunk_hits = []
    try:
        chunk_hits = await global_config["_chunks_vdb"].query(query, top_k=4)
    except Exception:
        pass
    ctx = _build_context(ent_hits, rels, chunk_hits, text_chunks, param)
    return await _generate(query, ctx, param, global_config)


async def hyper_query_lite(query, hg, ent_vdb, text_chunks, param, global_config):
    ent_hits = await _seed_entities(query, ent_vdb, param.top_k)
    chunk_hits = []
    try:
        chunk_hits = await global_config["_chunks_vdb"].query(query, top_k=4)
    except Exception:
        pass
    ctx = _build_context(ent_hits, [], chunk_hits, text_chunks, param)
    return await _generate(query, ctx, param, global_config)


async def graph_query(query, hg, ent_vdb, rel_vdb, text_chunks, param, global_config):
    ent_hits = await _seed_entities(query, ent_vdb, param.top_k)
    seeds = [_entity_name_of(h) for h in ent_hits]
    rels = await _diffuse(seeds, hg, include_high=False)  # pairwise only
    chunk_hits = []
    try:
        chunk_hits = await global_config["_chunks_vdb"].query(query, top_k=4)
    except Exception:
        pass
    ctx = _build_context(ent_hits, rels, chunk_hits, text_chunks, param)
    return await _generate(query, ctx, param, global_config)


async def naive_query(query, chunks_vdb, text_chunks, param, global_config):
    try:
        hits = await chunks_vdb.query(query, top_k=4)
    except Exception:
        hits = []
    ctx = _build_context([], [], hits, text_chunks, param)
    return await _generate(query, ctx, param, global_config)


async def llm_query(query, param, global_config):
    return await _generate(query, "(no retrieval — direct LLM answer)", param, global_config)


# ---------------- streaming variants ----------------

async def _generate_stream(query, context, param, global_config):
    stream_func = global_config.get("llm_model_stream_func")
    if stream_func is None or param.only_need_context:
        yield context if param.only_need_context else await _generate(query, context, param, global_config)
        return
    sys_p = QA_SYSTEM_PROMPT.format(response_type=param.response_type)
    user_p = QA_USER_TEMPLATE.format(context=context[:12000], question=query)
    async for tok in stream_func(user_p, system_prompt=sys_p):
        yield tok


async def hyper_query_stream(query, hg, ent_vdb, rel_vdb, text_chunks, param, global_config):
    ent_hits = await _seed_entities(query, ent_vdb, param.top_k)
    rels = await _diffuse([_entity_name_of(h) for h in ent_hits], hg, True)
    chunk_hits = await global_config["_chunks_vdb"].query(query, top_k=4) if "_chunks_vdb" in global_config else []
    async for t in _generate_stream(query, _build_context(ent_hits, rels, chunk_hits, text_chunks, param), param, global_config):
        yield t


async def hyper_query_lite_stream(query, hg, ent_vdb, text_chunks, param, global_config):
    ent_hits = await _seed_entities(query, ent_vdb, param.top_k)
    chunk_hits = await global_config["_chunks_vdb"].query(query, top_k=4) if "_chunks_vdb" in global_config else []
    async for t in _generate_stream(query, _build_context(ent_hits, [], chunk_hits, text_chunks, param), param, global_config):
        yield t


async def naive_query_stream(query, chunks_vdb, text_chunks, param, global_config):
    hits = await chunks_vdb.query(query, top_k=4)
    async for t in _generate_stream(query, _build_context([], [], hits, text_chunks, param), param, global_config):
        yield t


async def llm_query_stream(query, param, global_config):
    async for t in _generate_stream(query, "(no retrieval — direct LLM answer)", param, global_config):
        yield t
