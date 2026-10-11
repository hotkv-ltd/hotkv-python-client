# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""AI agent memory (AMEM.* and AMEM.SUMMARY.*), embedding dedup
(EMBED.DEDUP.*) and the retrieval-augmented-generation pipeline (RAG.*).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .._numbers import format_float, format_vector
from .._typing import KeyT, ValueT
from ._shared import (
    CallbackMap,
    ReadonlySet,
    as_float,
    cb_bool,
    cb_ok,
    expiry_args,
    force_text,
    named_fields,
    pairs_to_dict,
)

__all__ = ["AIMemoryCommands", "CALLBACKS", "READONLY"]


def _labeled(item: Sequence[Any]) -> Dict[str, Any]:
    """`["role", r, "content", c]` -> `{"role": r, "content": c}`; the label
    tokens are always force-decoded (protocol schema, not user data)."""
    return {force_text(k): v for k, v in pairs_to_dict(item).items()}


def _cb_turns(reply: Any, **_options: Any) -> List[Dict[str, Any]]:
    return [_labeled(turn) for turn in reply]


def _cb_recall(reply: Any, **_options: Any) -> List[Dict[str, Any]]:
    results = []
    for item in reply:
        d = _labeled(item)
        if "similarity" in d:
            d["similarity"] = as_float(d["similarity"])
        results.append(d)
    return results


def _cb_episodes(reply: Any, **_options: Any) -> List[Dict[str, Any]]:
    results = []
    for item in reply:
        d = _labeled(item)
        if "timestamp" in d:
            d["timestamp"] = int(d["timestamp"])
        results.append(d)
    return results


def _cb_summary_get(reply: Any, **_options: Any) -> Optional[Tuple[Any, int, int]]:
    if reply is None:
        return None
    summary, version, updated_unix = reply
    return (summary, int(version), int(updated_unix))


def _cb_rag_search(reply: Any, **_options: Any) -> List[Dict[str, Any]]:
    results = []
    for item in reply:
        chunk_id, doc_id, text, score, vector_score, bm25_score = item[:6]
        meta = {force_text(k): v for k, v in pairs_to_dict(item[6:]).items()}
        results.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": text,
                "score": as_float(score),
                "vector_score": as_float(vector_score),
                "bm25_score": as_float(bm25_score),
                "meta": meta,
            }
        )
    return results


CALLBACKS: CallbackMap = {
    "AMEM.CONV.CREATE": cb_ok,
    "AMEM.CONV.APPEND": cb_ok,
    "AMEM.CONV.GET": _cb_turns,
    "AMEM.CONV.WINDOW": _cb_turns,
    "AMEM.CONV.CLEAR": cb_bool,
    "AMEM.CONV.DROP": cb_bool,
    "AMEM.SEM.CREATE": cb_ok,
    "AMEM.SEM.STORE": cb_ok,
    "AMEM.SEM.RECALL": _cb_recall,
    "AMEM.SEM.FORGET": cb_bool,
    "AMEM.WORK.SET": cb_ok,
    "AMEM.WORK.DEL": cb_bool,
    "AMEM.WORK.CLEAR": cb_bool,
    "AMEM.EPIS.LOG": cb_ok,
    "AMEM.EPIS.QUERY": _cb_episodes,
    "AMEM.EPIS.CLEAR": cb_bool,
    "AMEM.EPIS.STATS": lambda reply, **_: named_fields(reply),
    "AMEM.SEARCH": _cb_recall,
    "AMEM.SUMMARY.GET": _cb_summary_get,
    "AMEM.SUMMARY.DROP": cb_bool,
    "EMBED.DEDUP.PUT": cb_bool,
    "EMBED.DEDUP.RELEASE": cb_bool,
    "EMBED.DEDUP.STATS": lambda reply, **_: named_fields(reply),
    "EMBED.DEDUP.DROP": cb_bool,
    "EMBED.DEDUP.PURGE": cb_bool,
    "RAG.CREATE": cb_ok,
    "RAG.SEARCH": _cb_rag_search,
    "RAG.STATS": lambda reply, **_: named_fields(reply, texts={"name", "chunkStrategy"}),
    "RAG.DROP": cb_ok,
}

READONLY: ReadonlySet = frozenset(
    {
        "AMEM.CONV.GET",
        "AMEM.CONV.WINDOW",
        "AMEM.SEM.RECALL",
        "AMEM.SEM.LIST",
        "AMEM.WORK.GET",
        "AMEM.WORK.LIST",
        "AMEM.EPIS.QUERY",
        "AMEM.EPIS.STATS",
        "AMEM.SEARCH",
        "AMEM.SUMMARY.GET",
        "AMEM.SUMMARY.LIST",
        "EMBED.DEDUP.GET",
        "EMBED.DEDUP.STATS",
        "EMBED.DEDUP.LIST",
        "RAG.SEARCH",
        "RAG.STATS",
        "RAG.LIST",
    }
)


class AIMemoryCommands:
    """Typed wrappers for AMEM.* (conversation/semantic/working/episodic
    agent memory), AMEM.SUMMARY.*, EMBED.DEDUP.* and RAG.*."""

    # -- AMEM.CONV.* (conversation memory) ---------------------------------------

    def amem_conv_create(
        self,
        session_id: ValueT,
        *,
        window: Optional[int] = None,
        ttl: Optional[int] = None,
        pxat: Optional[int] = None,
    ) -> bool:
        """Create conversation session `session_id`. `ttl` is seconds from now,
        `pxat` the Unix time in milliseconds the session ends at (not both)."""
        args: List[Any] = [session_id]
        if window is not None:
            args += ["WINDOW", window]
        args += expiry_args(ttl, pxat)
        return self.execute_command("AMEM.CONV.CREATE", *args)

    def amem_conv_append(self, session_id: ValueT, role: ValueT, content: ValueT) -> bool:
        """Append one turn to `session_id`."""
        return self.execute_command("AMEM.CONV.APPEND", session_id, role, content)

    def amem_conv_get(self, session_id: ValueT) -> List[Dict[str, Any]]:
        """Return every turn as `[{"role": ..., "content": ...}, ...]`."""
        return self.execute_command("AMEM.CONV.GET", session_id)

    def amem_conv_window(self, session_id: ValueT, n: int) -> List[Dict[str, Any]]:
        """Return the last `n` turns, same shape as `amem_conv_get()`."""
        return self.execute_command("AMEM.CONV.WINDOW", session_id, n)

    def amem_conv_clear(self, session_id: ValueT) -> bool:
        """Clear all turns from `session_id`, keeping the session."""
        return self.execute_command("AMEM.CONV.CLEAR", session_id)

    def amem_conv_drop(self, session_id: ValueT) -> bool:
        """Delete conversation session `session_id` entirely."""
        return self.execute_command("AMEM.CONV.DROP", session_id)

    # -- AMEM.SEM.* (semantic, per-agent vector memory) --------------------------

    def amem_sem_create(self, agent_id: ValueT, dimensions: int) -> bool:
        """Create a semantic memory store for `agent_id`."""
        return self.execute_command("AMEM.SEM.CREATE", agent_id, dimensions)

    def amem_sem_store(
        self,
        agent_id: ValueT,
        key: KeyT,
        content: ValueT,
        vector: Sequence[float],
        *,
        ttl: Optional[int] = None,
        pxat: Optional[int] = None,
    ) -> bool:
        """Store `content` under `key`, indexed by `vector`. `ttl` is seconds
        from now, `pxat` the Unix time in milliseconds it expires at (not both)."""
        args: List[Any] = [agent_id, key, content, *format_vector(vector)]
        args += expiry_args(ttl, pxat)
        return self.execute_command("AMEM.SEM.STORE", *args)

    def amem_sem_recall(
        self, agent_id: ValueT, k: int, vector: Sequence[float]
    ) -> List[Dict[str, Any]]:
        """Return up to `k` nearest memories to `vector`, most similar first."""
        return self.execute_command("AMEM.SEM.RECALL", agent_id, k, *format_vector(vector))

    def amem_sem_forget(self, agent_id: ValueT, key: KeyT) -> bool:
        """Remove one semantic memory by `key`."""
        return self.execute_command("AMEM.SEM.FORGET", agent_id, key)

    def amem_sem_list(self, agent_id: ValueT) -> List[bytes]:
        """List semantic memory keys for `agent_id`."""
        return self.execute_command("AMEM.SEM.LIST", agent_id)

    # -- AMEM.WORK.* (per-session scratch key/value) ----------------------------

    def amem_work_set(
        self,
        session_id: ValueT,
        key: KeyT,
        value: ValueT,
        *,
        ttl: Optional[int] = None,
        pxat: Optional[int] = None,
    ) -> bool:
        """Set scratch key `key` for `session_id`. `ttl` is seconds from now,
        `pxat` the Unix time in milliseconds it expires at (not both)."""
        args: List[Any] = [session_id, key, value]
        args += expiry_args(ttl, pxat)
        return self.execute_command("AMEM.WORK.SET", *args)

    def amem_work_get(self, session_id: ValueT, key: KeyT) -> Optional[bytes]:
        """Return scratch key `key`. Note: an empty stored value and a miss
        are indistinguishable (both reply `None`), per the server."""
        return self.execute_command("AMEM.WORK.GET", session_id, key)

    def amem_work_del(self, session_id: ValueT, key: KeyT) -> bool:
        """Delete scratch key `key`."""
        return self.execute_command("AMEM.WORK.DEL", session_id, key)

    def amem_work_clear(self, session_id: ValueT) -> bool:
        """Clear all scratch keys for `session_id`."""
        return self.execute_command("AMEM.WORK.CLEAR", session_id)

    def amem_work_list(self, session_id: ValueT) -> List[bytes]:
        """List scratch keys for `session_id`."""
        return self.execute_command("AMEM.WORK.LIST", session_id)

    # -- AMEM.EPIS.* (append-only event log) -------------------------------------

    def amem_epis_log(
        self,
        agent_id: ValueT,
        event_type: ValueT,
        content: ValueT,
        *,
        metadata: Optional[str] = None,
    ) -> bool:
        """Append an episodic event for `agent_id`."""
        args: List[Any] = [agent_id, event_type, content]
        if metadata is not None:
            args += ["METADATA", metadata]
        return self.execute_command("AMEM.EPIS.LOG", *args)

    def amem_epis_query(
        self,
        agent_id: ValueT,
        *,
        type: Optional[ValueT] = None,
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query episodic events for `agent_id`, newest filters optional."""
        args: List[Any] = [agent_id]
        if type is not None:
            args += ["TYPE", type]
        if from_ts is not None:
            args += ["FROM", from_ts]
        if to_ts is not None:
            args += ["TO", to_ts]
        if limit is not None:
            args += ["LIMIT", limit]
        return self.execute_command("AMEM.EPIS.QUERY", *args)

    def amem_epis_clear(self, agent_id: ValueT) -> bool:
        """Clear the episodic log for `agent_id`."""
        return self.execute_command("AMEM.EPIS.CLEAR", agent_id)

    def amem_epis_stats(self, agent_id: ValueT) -> Dict[str, int]:
        """Return `total_events`/`oldest_ts`/`newest_ts` for `agent_id`."""
        return self.execute_command("AMEM.EPIS.STATS", agent_id)

    # -- AMEM.SEARCH (semantic + episodic + working, combined) -------------------

    def amem_search(
        self, agent_id: ValueT, k: int, vector: Sequence[float]
    ) -> List[Dict[str, Any]]:
        """Search semantic, episodic and working memory together for `agent_id`."""
        return self.execute_command("AMEM.SEARCH", agent_id, k, *format_vector(vector))

    # -- AMEM.SUMMARY.* (requires a full license) --------------------------------

    def amem_summary_set(self, agent_id: ValueT, summary: ValueT) -> int:
        """Replace the rolling summary for `agent_id`, returning its new version."""
        return self.execute_command("AMEM.SUMMARY.SET", agent_id, summary)

    def amem_summary_append(self, agent_id: ValueT, extra: ValueT) -> int:
        """Append to the rolling summary for `agent_id`, returning its new version."""
        return self.execute_command("AMEM.SUMMARY.APPEND", agent_id, extra)

    def amem_summary_get(self, agent_id: ValueT) -> Optional[Tuple[Any, int, int]]:
        """Return `(summary, version, updated_unix)`, or `None`."""
        return self.execute_command("AMEM.SUMMARY.GET", agent_id)

    def amem_summary_drop(self, agent_id: ValueT) -> bool:
        """Delete the rolling summary for `agent_id`."""
        return self.execute_command("AMEM.SUMMARY.DROP", agent_id)

    def amem_summary_list(self) -> List[bytes]:
        """List agent IDs that have a stored summary."""
        return self.execute_command("AMEM.SUMMARY.LIST")

    # -- EMBED.DEDUP.* (requires a full license) ----------------------------------

    def embed_dedup_put(self, ns: ValueT, hash: ValueT, model: ValueT, embedding: ValueT) -> bool:
        """Store `embedding` under content `hash`; `True` if newly stored."""
        return self.execute_command("EMBED.DEDUP.PUT", ns, hash, model, embedding)

    def embed_dedup_get(self, ns: ValueT, hash: ValueT) -> Optional[bytes]:
        """Return the embedding stored for `hash`, or `None`."""
        return self.execute_command("EMBED.DEDUP.GET", ns, hash)

    def embed_dedup_release(self, ns: ValueT, hash: ValueT) -> bool:
        """Decrement the refcount for `hash` (freed at zero)."""
        return self.execute_command("EMBED.DEDUP.RELEASE", ns, hash)

    def embed_dedup_stats(self, ns: ValueT) -> Dict[str, int]:
        """Return `hits`/`misses`/`entries` for `ns`."""
        return self.execute_command("EMBED.DEDUP.STATS", ns)

    def embed_dedup_drop(self, ns: ValueT) -> bool:
        """Drop dedup namespace `ns`."""
        return self.execute_command("EMBED.DEDUP.DROP", ns)

    def embed_dedup_purge(self, ns: ValueT) -> bool:
        """Purge expired/unreferenced entries in `ns`."""
        return self.execute_command("EMBED.DEDUP.PURGE", ns)

    def embed_dedup_list(self) -> List[bytes]:
        """List dedup namespace names."""
        return self.execute_command("EMBED.DEDUP.LIST")

    # -- RAG.* (retrieval-augmented-generation pipeline) --------------------------

    def rag_create(
        self,
        name: ValueT,
        dimensions: int,
        *,
        strategy: str = "fixed",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> bool:
        """Create RAG pipeline `name`."""
        args: List[Any] = [name, dimensions, "STRATEGY", strategy]
        if chunk_size is not None:
            args += ["CHUNK_SIZE", chunk_size]
        if chunk_overlap is not None:
            args += ["CHUNK_OVERLAP", chunk_overlap]
        return self.execute_command("RAG.CREATE", *args)

    def rag_ingest(
        self,
        pipeline: ValueT,
        doc_id: ValueT,
        text: ValueT,
        *,
        meta: Optional[Mapping[str, ValueT]] = None,
        vectors: Optional[Sequence[Sequence[float]]] = None,
    ) -> int:
        """Ingest `text` under `doc_id`, chunking it server side.

        `vectors`, if given, supplies one embedding per chunk the text will
        be split into; each is a sequence of floats, comma-joined on the wire.
        """
        args: List[Any] = [pipeline, doc_id, text]
        if meta:
            args.append("META")
            for key, value in meta.items():
                args += [key, value]
        if vectors:
            args.append("VECTORS")
            for vector in vectors:
                args.append(",".join(format_vector(vector)))
        return self.execute_command("RAG.INGEST", *args)

    def rag_search(
        self,
        pipeline: ValueT,
        query_text: ValueT,
        *,
        k: Optional[int] = None,
        alpha: Optional[float] = None,
        vector: Optional[Sequence[float]] = None,
        filter: Optional[Mapping[str, ValueT]] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid (vector + BM25) search; `alpha` blends 0 (pure BM25) to 1
        (pure vector). `filter` must be passed last on the wire, so build it
        after `k`/`alpha`/`vector` regardless of keyword order given here."""
        args: List[Any] = [pipeline, query_text]
        if k is not None:
            args += ["K", k]
        if alpha is not None:
            args += ["ALPHA", format_float(alpha)]
        if vector is not None:
            args += ["VECTOR", ",".join(format_vector(vector))]
        if filter:
            args.append("FILTER")
            for key, value in filter.items():
                args += [key, value]
        return self.execute_command("RAG.SEARCH", *args)

    def rag_stats(self, pipeline: ValueT) -> Dict[str, Any]:
        """Return document/chunk/query counts and chunking config for `pipeline`."""
        return self.execute_command("RAG.STATS", pipeline)

    def rag_drop(self, pipeline: ValueT) -> bool:
        """Delete RAG pipeline `pipeline`."""
        return self.execute_command("RAG.DROP", pipeline)

    def rag_list(self) -> List[bytes]:
        """List RAG pipeline names."""
        return self.execute_command("RAG.LIST")
