"""Token-budgeted RAG context assembly."""

from __future__ import annotations

import sqlite3

from rememble.config import RAGConfig, SearchConfig
from rememble.ingest.chunker import countTokens, truncateToTokens
from rememble.models import RAGContext, RAGItem
from rememble.search.fusion import hybridSearch


def buildContext(
    db: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    search_config: SearchConfig,
    rag_config: RAGConfig,
) -> RAGContext:
    """Build token-budgeted context from hybrid search results.

    Pipeline:
    1. Hybrid search (overfetch: limit * 3)
    2. Expansion: full text of top result (capped at expansion_max_tokens)
    3. Snippets: preview text from remaining results (snippet_max_tokens each)
    4. Graph context: entity/observation text for matched entities
    5. Token budget: greedily fill until max_context_tokens
    """
    # Overfetch for better candidate pool
    search_result = hybridSearch(
        db,
        query,
        query_embedding,
        search_config,
        limit=rag_config.max_snippets * 3,
    )

    items: list[RAGItem] = []
    remaining_tokens = rag_config.max_context_tokens

    # Step 1: Expansion — full text of top result
    if search_result.results:
        top = search_result.results[0]
        if top.content:
            expanded_text = truncateToTokens(top.content, rag_config.expansion_max_tokens)
            tokens = countTokens(expanded_text)
            if tokens <= remaining_tokens:
                items.append(
                    RAGItem(
                        kind="expanded",
                        memory_id=top.memory_id,
                        score=top.score,
                        text=expanded_text,
                        tokens=tokens,
                    )
                )
                remaining_tokens -= tokens

    # Step 2: Snippets from remaining results
    seen_ids = {item.memory_id for item in items}
    snippet_count = 0

    for result in search_result.results:
        if remaining_tokens <= 0 or snippet_count >= rag_config.max_snippets:
            break
        if result.memory_id in seen_ids:
            continue
        seen_ids.add(result.memory_id)

        # Use snippet if available, else truncate content
        text = result.snippet or (result.content or "")
        if not text.strip():
            continue

        text = truncateToTokens(text, rag_config.snippet_max_tokens)
        tokens = countTokens(text)
        if tokens > remaining_tokens:
            text = truncateToTokens(text, remaining_tokens)
            tokens = countTokens(text)

        items.append(
            RAGItem(
                kind="snippet",
                memory_id=result.memory_id,
                score=result.score,
                text=text,
                tokens=tokens,
            )
        )
        remaining_tokens -= tokens
        snippet_count += 1

    # Step 3: Graph context — entity observations as additional context
    for gr in search_result.graph:
        if remaining_tokens <= 0:
            break
        obs_texts = [o.content for o in gr.observations]
        rel_texts = [
            f"{gr.entity.name} {rwe.relation.relation_type} {rwe.entity.name}"
            for rwe in gr.relations
            if rwe.direction == "outbound"
        ]
        graph_text = f"Entity: {gr.entity.name} ({gr.entity.entity_type})\n"
        if obs_texts:
            graph_text += "Facts: " + "; ".join(obs_texts) + "\n"
        if rel_texts:
            graph_text += "Relations: " + "; ".join(rel_texts)

        max_graph_tokens = min(rag_config.snippet_max_tokens, remaining_tokens)
        graph_text = truncateToTokens(graph_text.strip(), max_graph_tokens)
        tokens = countTokens(graph_text)

        items.append(
            RAGItem(
                kind="graph",
                score=0.0,
                text=graph_text,
                tokens=tokens,
            )
        )
        remaining_tokens -= tokens

    total_tokens = sum(item.tokens for item in items)

    return RAGContext(
        query=query,
        items=items,
        total_tokens=total_tokens,
        entities=search_result.graph,
    )
