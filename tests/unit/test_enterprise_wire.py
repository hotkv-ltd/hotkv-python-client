# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""The exact wire arguments of the typed methods of the enterprise engines.

An unlicensed server answers NOLICENSE to these commands, so they cannot be run
end to end without a license. Each row is a call and the argument list the
command takes for it: command name, then the arguments in the order given,
keyword options as KEY VALUE pairs.
Reply shaping is not involved; see test_ai_cache_commands.py for that. No
server needed."""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from hotkv import TokenUsage, commands


class Wire(commands.CoreCommands):
    """Records the arguments of the one command a call sends."""

    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []

    def execute_command(self, *args: Any, **options: Any) -> Any:
        self.calls.append(args)
        return None


U = TokenUsage(model="m", input_tokens=5, output_tokens=2)

ROWS = [
    # -- AMEM.CONV.* / SEM / WORK / EPIS / SEARCH ---------------------------------
    (lambda c: c.amem_conv_append("s", "user", "hi"), ("AMEM.CONV.APPEND", "s", "user", "hi")),
    (lambda c: c.amem_conv_get("s"), ("AMEM.CONV.GET", "s")),
    (lambda c: c.amem_conv_window("s", 3), ("AMEM.CONV.WINDOW", "s", 3)),
    (lambda c: c.amem_conv_clear("s"), ("AMEM.CONV.CLEAR", "s")),
    (lambda c: c.amem_conv_drop("s"), ("AMEM.CONV.DROP", "s")),
    (lambda c: c.amem_conv_create("s"), ("AMEM.CONV.CREATE", "s")),
    (lambda c: c.amem_conv_create("s", window=4, ttl=60), ("AMEM.CONV.CREATE", "s", "WINDOW", 4, "TTL", 60)),
    (lambda c: c.amem_sem_create("a", 3), ("AMEM.SEM.CREATE", "a", 3)),
    (lambda c: c.amem_sem_store("a", "k", "text", [0.5, 0.25], ttl=9), ("AMEM.SEM.STORE", "a", "k", "text", "0.5", "0.25", "TTL", 9)),
    (lambda c: c.amem_sem_recall("a", 2, [1, 0]), ("AMEM.SEM.RECALL", "a", 2, "1", "0")),
    (lambda c: c.amem_sem_forget("a", "k"), ("AMEM.SEM.FORGET", "a", "k")),
    (lambda c: c.amem_sem_list("a"), ("AMEM.SEM.LIST", "a")),
    (lambda c: c.amem_work_set("s", "k", "v", ttl=5), ("AMEM.WORK.SET", "s", "k", "v", "TTL", 5)),
    (lambda c: c.amem_work_get("s", "k"), ("AMEM.WORK.GET", "s", "k")),
    (lambda c: c.amem_work_del("s", "k"), ("AMEM.WORK.DEL", "s", "k")),
    (lambda c: c.amem_work_clear("s"), ("AMEM.WORK.CLEAR", "s")),
    (lambda c: c.amem_work_list("s"), ("AMEM.WORK.LIST", "s")),
    (lambda c: c.amem_epis_log("a", "tool", "ran"), ("AMEM.EPIS.LOG", "a", "tool", "ran")),
    (lambda c: c.amem_epis_log("a", "tool", "ran", metadata="{}"), ("AMEM.EPIS.LOG", "a", "tool", "ran", "METADATA", "{}")),
    (
        lambda c: c.amem_epis_query("a", type="tool", from_ts=1, to_ts=9, limit=3),
        ("AMEM.EPIS.QUERY", "a", "TYPE", "tool", "FROM", 1, "TO", 9, "LIMIT", 3),
    ),
    (lambda c: c.amem_epis_query("a"), ("AMEM.EPIS.QUERY", "a")),
    (lambda c: c.amem_epis_clear("a"), ("AMEM.EPIS.CLEAR", "a")),
    (lambda c: c.amem_epis_stats("a"), ("AMEM.EPIS.STATS", "a")),
    (lambda c: c.amem_search("a", 5, [0.5, 1.5]), ("AMEM.SEARCH", "a", 5, "0.5", "1.5")),
    (lambda c: c.amem_summary_set("a", "text"), ("AMEM.SUMMARY.SET", "a", "text")),
    (lambda c: c.amem_summary_append("a", "more"), ("AMEM.SUMMARY.APPEND", "a", "more")),
    (lambda c: c.amem_summary_get("a"), ("AMEM.SUMMARY.GET", "a")),
    (lambda c: c.amem_summary_drop("a"), ("AMEM.SUMMARY.DROP", "a")),
    (lambda c: c.amem_summary_list(), ("AMEM.SUMMARY.LIST",)),
    # -- EMBED.DEDUP.* -------------------------------------------------------------
    (lambda c: c.embed_dedup_put("ns", "h", "model", "1,2"), ("EMBED.DEDUP.PUT", "ns", "h", "model", "1,2")),
    (lambda c: c.embed_dedup_get("ns", "h"), ("EMBED.DEDUP.GET", "ns", "h")),
    (lambda c: c.embed_dedup_release("ns", "h"), ("EMBED.DEDUP.RELEASE", "ns", "h")),
    (lambda c: c.embed_dedup_stats("ns"), ("EMBED.DEDUP.STATS", "ns")),
    (lambda c: c.embed_dedup_drop("ns"), ("EMBED.DEDUP.DROP", "ns")),
    (lambda c: c.embed_dedup_purge("ns"), ("EMBED.DEDUP.PURGE", "ns")),
    (lambda c: c.embed_dedup_list(), ("EMBED.DEDUP.LIST",)),
    # -- RAG.* ----------------------------------------------------------------------
    (lambda c: c.rag_create("p", 8), ("RAG.CREATE", "p", 8, "STRATEGY", "fixed")),
    (
        lambda c: c.rag_create("p", 8, strategy="sentence", chunk_size=200, chunk_overlap=20),
        ("RAG.CREATE", "p", 8, "STRATEGY", "sentence", "CHUNK_SIZE", 200, "CHUNK_OVERLAP", 20),
    ),
    (lambda c: c.rag_ingest("p", "d", "text", meta={"a": "b"}), ("RAG.INGEST", "p", "d", "text", "META", "a", "b")),
    (lambda c: c.rag_search("p", "q"), ("RAG.SEARCH", "p", "q")),
    (
        lambda c: c.rag_search("p", "q", k=3, alpha=0.5, vector=[1, 2], filter={"x": "y"}),
        ("RAG.SEARCH", "p", "q", "K", 3, "ALPHA", "0.5", "VECTOR", "1,2", "FILTER", "x", "y"),
    ),
    (lambda c: c.rag_stats("p"), ("RAG.STATS", "p")),
    (lambda c: c.rag_drop("p"), ("RAG.DROP", "p")),
    (lambda c: c.rag_list(), ("RAG.LIST",)),
    # -- RATELIMIT.* -----------------------------------------------------------------
    (lambda c: c.ratelimit_create("r", 10, 1000), ("RATELIMIT.CREATE", "r", 10, 1000, "ALGO", "sliding_window")),
    (lambda c: c.ratelimit_check("r", "k"), ("RATELIMIT.CHECK", "r", "k")),
    (lambda c: c.ratelimit_check("r", "k", cost=3), ("RATELIMIT.CHECK", "r", "k", "COST", 3)),
    (lambda c: c.ratelimit_peek("r", "k"), ("RATELIMIT.PEEK", "r", "k")),
    (lambda c: c.ratelimit_reset("r", "k"), ("RATELIMIT.RESET", "r", "k")),
    (lambda c: c.ratelimit_stats("r"), ("RATELIMIT.STATS", "r")),
    (lambda c: c.ratelimit_drop("r"), ("RATELIMIT.DROP", "r")),
    (lambda c: c.ratelimit_list(), ("RATELIMIT.LIST",)),
    # -- TS.* --------------------------------------------------------------------------
    (lambda c: c.ts_create("t"), ("TS.CREATE", "t")),
    (lambda c: c.ts_create("t", retention_ms=1000, labels={"a": "b"}), ("TS.CREATE", "t", "RETENTION", 1000, "LABELS", "a", "b")),
    (lambda c: c.ts_add("t", 2.5, 1000), ("TS.ADD", "t", 1000, "2.5")),
    (lambda c: c.ts_range("t", "-", "+", count=5), ("TS.RANGE", "t", "-", "+", "COUNT", 5)),
    (lambda c: c.ts_agg("t", 0, 100, 10, "avg"), ("TS.AGG", "t", 0, 100, 10, "avg")),
    (lambda c: c.ts_info("t"), ("TS.INFO", "t")),
    (lambda c: c.ts_delete("t"), ("TS.DELETE", "t")),
    (lambda c: c.ts_retention("t", 500), ("TS.RETENTION", "t", 500)),
    (lambda c: c.ts_downsample("src", "dst", "avg", 60000), ("TS.DOWNSAMPLE", "src", "dst", "avg", 60000)),
    (lambda c: c.ts_list(), ("TS.LIST",)),
    (lambda c: c.ts_list(label=("a", "b")), ("TS.LIST", "LABEL", "a", "b")),
    # -- FEATURE.* -----------------------------------------------------------------------
    (
        lambda c: c.feature_create("g", "desc", {"n": "int64", "v": "double"}),
        ("FEATURE.CREATE", "g", "desc", "SCHEMA", "n", "int64", "v", "double"),
    ),
    (lambda c: c.feature_set("g", "e", {"n": 1, "v": "2.5"}), ("FEATURE.SET", "g", "e", "n", 1, "v", "2.5")),
    (lambda c: c.feature_get("g", "e"), ("FEATURE.GET", "g", "e")),
    (lambda c: c.feature_get("g", "e", ["n"]), ("FEATURE.GET", "g", "e", "n")),
    (lambda c: c.feature_batch("g", ["e1", "e2"]), ("FEATURE.BATCH", "g", "e1", "e2")),
    (lambda c: c.feature_batch("g", ["e1"], features=["n", "v"]), ("FEATURE.BATCH", "g", "e1", "FEATURES", "n", "v")),
    (lambda c: c.feature_schema("g"), ("FEATURE.SCHEMA", "g")),
    (lambda c: c.feature_list(), ("FEATURE.LIST",)),
    (lambda c: c.feature_drop("g"), ("FEATURE.DROP", "g")),
    (lambda c: c.feature_stats("g"), ("FEATURE.STATS", "g")),
    # -- PCACHE.* / SCACHE.* ----------------------------------------------------------------
    (lambda c: c.pcache_create("ns"), ("PCACHE.CREATE", "ns")),
    (
        lambda c: c.pcache_create("ns", max_entries=100, ttl=60, cost_per_hit=0.002),
        ("PCACHE.CREATE", "ns", "MAX_ENTRIES", 100, "TTL", 60, "COST_PER_HIT", "0.002"),
    ),
    (lambda c: c.pcache_set("ns", "p", "r", meta="{}", usage=U), ("PCACHE.SET", "ns", "p", "r", "META", "{}", "MODEL", "m", "INPUT", 5, "OUTPUT", 2)),
    (lambda c: c.pcache_get("ns", "p"), ("PCACHE.GET", "ns", "p")),
    (lambda c: c.pcache_getex("ns", "p"), ("PCACHE.GETEX", "ns", "p")),
    (lambda c: c.pcache_del("ns", "p"), ("PCACHE.DEL", "ns", "p")),
    (lambda c: c.pcache_stats("ns"), ("PCACHE.STATS", "ns")),
    (lambda c: c.pcache_flush("ns"), ("PCACHE.FLUSH", "ns")),
    (lambda c: c.pcache_drop("ns"), ("PCACHE.DROP", "ns")),
    (lambda c: c.pcache_list(), ("PCACHE.LIST",)),
    (lambda c: c.pcache_config_set("ns", "ttl", 60), ("PCACHE.CONFIG", "ns", "SET", "TTL", 60)),
    (lambda c: c.pcache_config_set("ns", "max_entries", 9), ("PCACHE.CONFIG", "ns", "SET", "MAX_ENTRIES", 9)),
    (lambda c: c.scache_create("ns", 4), ("SCACHE.CREATE", "ns", 4, "METRIC", "cosine", "THRESHOLD", "0.85")),
    (lambda c: c.scache_create("ns", 4, metric="l2", threshold=0.5, ttl=30), ("SCACHE.CREATE", "ns", 4, "METRIC", "l2", "THRESHOLD", "0.5", "TTL", 30)),
    (lambda c: c.scache_get("ns", [1, 0]), ("SCACHE.GET", "ns", "1", "0")),
    (lambda c: c.scache_stats("ns"), ("SCACHE.STATS", "ns")),
    (lambda c: c.scache_flush("ns"), ("SCACHE.FLUSH", "ns")),
    (lambda c: c.scache_drop("ns"), ("SCACHE.DROP", "ns")),
    (lambda c: c.scache_list(), ("SCACHE.LIST",)),
    # -- LLMSTATS.* / LLMBUDGET.* / PROMPT.* ---------------------------------------------------
    (lambda c: c.llmstats_bucket("b"), ("LLMSTATS.BUCKET", "b")),
    (lambda c: c.llmstats_get("b"), ("LLMSTATS.GET", "b")),
    (lambda c: c.llmstats_reset("b"), ("LLMSTATS.RESET", "b")),
    (lambda c: c.llmstats_drop("b"), ("LLMSTATS.DROP", "b")),
    (lambda c: c.llmstats_list(), ("LLMSTATS.LIST",)),
    (lambda c: c.llmstats_price_get("m"), ("LLMSTATS.PRICE GET", "m")),
    (lambda c: c.llmstats_price_list(), ("LLMSTATS.PRICE LIST",)),
    (lambda c: c.llmstats_price_del("m"), ("LLMSTATS.PRICE DEL", "m")),
    (lambda c: c.llmbudget_status("b"), ("LLMBUDGET.STATUS", "b")),
    (lambda c: c.llmbudget_reset("b"), ("LLMBUDGET.RESET", "b")),
    (lambda c: c.llmbudget_drop("b"), ("LLMBUDGET.DROP", "b")),
    (lambda c: c.llmbudget_list(), ("LLMBUDGET.LIST",)),
    (lambda c: c.prompt_put("n", "content"), ("PROMPT.PUT", "n", "content")),
    (lambda c: c.prompt_put("n", "content", meta="{}"), ("PROMPT.PUT", "n", "content", "META", "{}")),
    (lambda c: c.prompt_get("n"), ("PROMPT.GET", "n")),
    (lambda c: c.prompt_get("n", 2), ("PROMPT.GET", "n", 2)),
    (lambda c: c.prompt_latest("n"), ("PROMPT.LATEST", "n")),
    (lambda c: c.prompt_setactive("n", 2), ("PROMPT.SETACTIVE", "n", 2)),
    (lambda c: c.prompt_versions("n"), ("PROMPT.VERSIONS", "n")),
    (lambda c: c.prompt_drop("n"), ("PROMPT.DROP", "n")),
    (lambda c: c.prompt_list(), ("PROMPT.LIST",)),
    # -- the keyed caches -------------------------------------------------------------------------
    *[
        row
        for prefix, name in (("TOOL.CACHE", "toolcache"), ("NCACHE", "ncache"), ("GUARD", "guard"), ("RERANK", "rerank"))
        for row in (
            (lambda c, n=name: getattr(c, f"{n}_create")("ns"), (f"{prefix}.CREATE", "ns")),
            (
                lambda c, n=name: getattr(c, f"{n}_create")("ns", max_entries=10, ttl=5),
                (f"{prefix}.CREATE", "ns", "MAX_ENTRIES", 10, "TTL", 5),
            ),
            (lambda c, n=name: getattr(c, f"{n}_set")("ns", "k", "v"), (f"{prefix}.SET", "ns", "k", "v")),
            (
                lambda c, n=name: getattr(c, f"{n}_set")("ns", "k", "v", ttl=5, meta="{}", usage=U),
                (f"{prefix}.SET", "ns", "k", "v", "TTL", 5, "META", "{}", "MODEL", "m", "INPUT", 5, "OUTPUT", 2),
            ),
            (lambda c, n=name: getattr(c, f"{n}_get")("ns", "k"), (f"{prefix}.GET", "ns", "k")),
            (lambda c, n=name: getattr(c, f"{n}_del")("ns", "k"), (f"{prefix}.DEL", "ns", "k")),
            (lambda c, n=name: getattr(c, f"{n}_stats")("ns"), (f"{prefix}.STATS", "ns")),
            (lambda c, n=name: getattr(c, f"{n}_flush")("ns"), (f"{prefix}.FLUSH", "ns")),
            (lambda c, n=name: getattr(c, f"{n}_drop")("ns"), (f"{prefix}.DROP", "ns")),
            (lambda c, n=name: getattr(c, f"{n}_list")(), (f"{prefix}.LIST",)),
        )
    ],
    # -- OBSERVE / ALERTS / VSIM snapshots ----------------------------------------------------------
    (lambda c: c.observe_info(), ("OBSERVE INFO",)),
    (lambda c: c.observe_metrics(), ("OBSERVE METRICS",)),
    (lambda c: c.alerts_info(), ("ALERTS INFO",)),
    (lambda c: c.alerts_setseverity("warning"), ("ALERTS SETSEVERITY", "warning")),
    (lambda c: c.alerts_enable(), ("ALERTS ON",)),
    (lambda c: c.alerts_disable(), ("ALERTS OFF",)),
    (lambda c: c.vsim_save("snap", b"\x00blob"), ("VSIM.SAVE", "snap", b"\x00blob")),
    (lambda c: c.vsim_load("snap"), ("VSIM.LOAD", "snap")),
    (lambda c: c.vsim_list(), ("VSIM.LIST",)),
    (lambda c: c.vsim_drop("snap"), ("VSIM.DROP", "snap")),
]


@pytest.mark.parametrize("call, expected", ROWS, ids=[r[1][0] + "-" + str(i) for i, r in enumerate(ROWS)])
def test_wire_arguments(call, expected):
    client = Wire()
    call(client)
    assert client.calls == [expected]


def test_feature_setat_arguments():
    client = Wire()
    client.feature_setat("g", "e", 1700000000000, {"n": 1, "v": "2.5"})
    assert client.calls == [("FEATURE.SETAT", "g", "e", 1700000000000, "n", 1, "v", "2.5")]
