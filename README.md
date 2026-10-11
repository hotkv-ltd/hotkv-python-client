# HotKV Python Client

[![CI](https://github.com/hotkv-ltd/hotkv-python-client/actions/workflows/ci.yml/badge.svg)](https://github.com/hotkv-ltd/hotkv-python-client/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A high-performance Python client for [HotKV](https://hotkv.com), a RESP2/RESP3-compatible
in-memory data store with built-in AI and LLM-serving features (semantic and prompt
caching, agent memory, RAG, feature store, rate limiting, time series and more).

- Sync and asyncio clients sharing one command layer
- Thread safe connection pooling, TLS, Unix sockets
- Pipelines, transactions (`MULTI`/`EXEC`/`WATCH`), pub/sub
- A slot-aware cluster client
- Typed methods for the data commands and for most HotKV exclusive and AI
  commands; any other command goes through `client.execute()` (see
  [What has a typed method](#what-has-a-typed-method))
- Zero runtime dependencies (standard library only), full type hints

Written against HotKV server 0.3.0; the tests run against that version.

## Install

```bash
pip install hotkv-client
```

The package is called `hotkv-client` on PyPI and imported as `hotkv`.

Requires Python 3.9 or later. The LangChain, LlamaIndex and LiteLLM
integrations (see below) need Python 3.10 or later, which those libraries
require; the rest of the client runs on 3.9.

## Quick start

```python
import hotkv

client = hotkv.HotKV(host="localhost", port=6379)
client.set("greeting", "hello")
print(client.get("greeting"))  # b"hello"
client.close()
```

Or build a client from a connection string:

```python
client = hotkv.from_url("hotkv://user:password@localhost:6379/0")
```

`hotkv.from_url()` and `HotKV(...)` accept `hotkv://`, `hotkvs://` (TLS),
and, for drop-in migration, `redis://`/`rediss://`; `unix:///path/to.sock`
addresses a Unix domain socket.

### Asyncio

```python
import asyncio
import hotkv.asyncio

async def main():
    client = hotkv.asyncio.HotKV(host="localhost", port=6379)
    await client.set("greeting", "hello")
    print(await client.get("greeting"))
    await client.close()

asyncio.run(main())
```

The asyncio client mirrors the sync one method for method. One important
difference: because its `execute_command()` is a coroutine function, every
call on an asyncio pipeline must be awaited too, even while queuing (see
Pipelines below).

## Connecting

```python
client = hotkv.HotKV(
    host="localhost",
    port=6379,
    username="app",
    password="secret",
    db=0,
    client_name="my-service",
    connect_timeout=5.0,
    command_timeout=10.0,
)
```

By default the client negotiates RESP3 with `HELLO 3` and falls back to
RESP2 automatically if the server rejects it. Set `decode_responses=True`
to get `str` back instead of `bytes` for text-like replies (binary values
such as `GET` on non-text data are always `bytes`, regardless of this
setting).

### TLS

```python
client = hotkv.HotKV(
    host="hotkv.example.com",
    port=6380,
    tls=True,
    tls_ca_file="/etc/ssl/certs/ca.pem",
    tls_certfile="/etc/ssl/certs/client.pem",
    tls_keyfile="/etc/ssl/private/client.key",
    tls_server_hostname="hotkv.example.com",
)
```

`tls_insecure_skip_verify=True` disables certificate verification; use it
only in tests, never in production. `hotkvs://` and `rediss://` URLs enable
TLS automatically.

### Unix sockets

```python
client = hotkv.HotKV(unix_socket_path="/var/run/hotkv/hotkv.sock")
# or
client = hotkv.from_url("unix:///var/run/hotkv/hotkv.sock?db=1")
```

### Connection pooling

Every `HotKV`/`hotkv.asyncio.HotKV` instance owns a bounded, thread safe
connection pool (`max_connections`, default 50). Connections are created
lazily and reused automatically; there is nothing to check out or return
by hand. An idle connection is health checked with `PING` before reuse
(`idle_timeout`, default 300 seconds) and replaced if it fails.

```python
client = hotkv.HotKV(max_connections=100, idle_timeout=60.0)
```

Reads that fail because a connection dropped are retried once automatically
(idempotent by nature); writes are not retried unless you opt in with
`retry_writes=True`, since blindly retrying a write can duplicate it.

## Pipelines

Batch several commands into one round trip:

```python
with client.pipeline(transaction=False) as pipe:
    pipe.set("a", 1)
    pipe.incr("counter")
    pipe.get("a")
    results = pipe.execute()  # [True, 2, b"1"], in order
```

`transaction=False` gives you pure pipelining (fastest, no atomicity).
`client.pipeline()` defaults to `transaction=True`, which wraps the same
batch in `MULTI`/`EXEC` for atomicity. Pass `raise_on_error=False` to
`execute()` to get a per-command `ResponseError` in the results list
instead of aborting the whole batch on the first failure.

Asyncio pipelines work the same way, but every call must be awaited, even
while queuing (queuing itself does no I/O, but `execute_command` is a
coroutine function either way):

```python
async with client.pipeline(transaction=False) as pipe:
    await pipe.set("a", 1)
    await pipe.incr("counter")
    results = await pipe.execute()
```

## Transactions

Low level `WATCH`/`MULTI`/`EXEC`:

```python
pipe = client.pipeline()
pipe.watch("balance")
current = pipe.get("balance")   # runs immediately, before MULTI
pipe.multi()                    # switch to queueing mode
pipe.set("balance", int(current) - 10)
pipe.execute()                  # MULTI + queued command(s) + EXEC
```

`EXEC` returns `None` if a watched key changed, which is surfaced as
`hotkv.WatchError`. The higher level `client.transaction()` retries for you:

```python
def debit(pipe):
    current = pipe.get("balance")
    pipe.multi()
    pipe.set("balance", int(current) - 10)

client.transaction(debit, "balance", max_retries=5)
```

## Pub/Sub

```python
pubsub = client.pubsub()
pubsub.subscribe("news")
for message in pubsub:
    print(message.channel, message.data)
```

`subscribe()`, `psubscribe()` (glob patterns) and `ssubscribe()` (shard
channels) are all supported, along with their `un*` counterparts.
`get_message(timeout=...)` polls without blocking forever;
`pubsub.listen()` (or iterating the object directly) blocks for each
message. The asyncio client exposes the same API as `async for message in
pubsub:` / `await pubsub.get_message(...)`.

## Cluster

```python
from hotkv.cluster import ClusterHotKV

cluster = ClusterHotKV([("10.0.0.1", 6379), ("10.0.0.2", 6379)])
cluster.set("user:1000", "value")
cluster.close()
```

The cluster client builds its slot map from `CLUSTER SLOTS`, hashes keys
with the same CRC16 plus `{hash tag}` scheme as `CLUSTER KEYSLOT`, follows
`MOVED` (refreshing the map) and `ASK` (sending `ASKING` first) redirects,
and can read from replicas with `read_from_replicas=True` (its replica
connections send `READONLY`). It supports every typed method for
single commands; pipelines, transactions and pub/sub are not supported on the
cluster client, since they need every key involved to share a slot.
`hotkv.asyncio.ClusterHotKV` is the asyncio equivalent.

It also rides out a live cluster. A lost primary (failover), a reshard in
flight, `CLUSTERDOWN`, `TRYAGAIN` and `LOADING` make a command refresh the
slot map and try again with a short backoff, for up to `retry_timeout`
seconds (default 10) before it raises `ClusterError`. A command that may have
reached a node whose connection then broke is sent again, as other cluster
clients do; give the client a `command_timeout` so a node that stops answering
is given up on. That is at-least-once delivery: when the first copy was in fact
applied, a command that is not idempotent (`incr`, `rpush`, `xadd`, `ts_add`,
`llmbudget_consume`, `ratelimit_check`, `llmstats_record`, `amem_conv_append` and
the other counters and appends) is applied twice. Pass `retry_writes=False` to
have such a failure raised to you instead (a read, and a command that never left
the client, are retried either way). A refused password or certificate is
raised at once, not retried. A command is
routed by its first key (`EVAL`'s and `XREAD`'s too, where it is not the first
argument), and commands that name several keys need them in one slot: give
them a common hash tag, as on any cluster.

Some commands the server answers from a node's own state, not from a slot.
The cluster client sends these to every primary and merges the answers, so
they cover the whole cluster: the `*_list()` commands of the AI engines
(`pcache_list()`, `scache_list()`, `vsim_info()` without an index,
`llmbudget_list()`, `ratelimit_list()`, ...), `tag_members()`/`tag_inter()`,
`keys()`, `dbsize()` and `flushall()`. So are `script_load()` and `script_flush()`
and the `FUNCTION LOAD`/`DELETE`/`FLUSH`/`RESTORE` commands (sent with
`execute()`): the script cache and the function libraries belong to a node, so
`eval_auto()` and `FCALL` find what was loaded through the client wherever the
key lives. `vsim_load()` and `vsim_drop()` ask every primary too, and
`vsim_save()` goes to the owner of the snapshot's name, so a snapshot that
`vsim_list()` shows is found (if a reshard left the same name on two nodes, the
first one found is returned). `scan_iter()` walks the keyspace of every primary;
`execute_on_node()` and `execute_on_primaries()` send a command to one node or
to each primary.

The commands that act on one node's own state (`ENCRYPTION.*`,
`AUDIT.*`, `OBSERVE`, `ALERTS`, `HOTKEYS`, `ACL`, `CONFIG`, `INFO`, `SLOWLOG`)
go to one primary, not to all of them: an ACL user exists on the nodes it was
created on and no others. Send them with `execute_on_node()` to the node you
mean, or with `execute_on_primaries()` to every primary.

Every AI engine entry (a cache namespace, a vector index, a rate limiter, a
budget, an agent's memory, ...) is keyed by its name, so it lives on one node,
and a name is all a command needs. See "What is clustered" in the HotKV server
README for what is sharded, what every node holds and what stays on one node.
The LangChain, LlamaIndex and LiteLLM integrations take a `ClusterHotKV` in
place of a `HotKV`.

## What has a typed method

Counts for server 0.3.0: 216 of the 444 core command names (the
sub-commands of `CONFIG`, `CLIENT`, `XINFO`, ... count one each) and 145 of the 177 enterprise ones have a typed method, and 17 more
core commands are built into the client (`MULTI`/`EXEC`/`WATCH`/`UNWATCH`/
`DISCARD` in pipelines, the `SUBSCRIBE` family in `pubsub()`, the `HELLO`,
`AUTH`, `SELECT` and `READONLY` handshake, `ASKING` in the cluster client).

Typed methods cover strings, keys and expiry, lists, sets, hashes (including
field TTLs), sorted sets, streams and consumer groups, geo, bitmaps,
HyperLogLog, scripting (`EVAL`, `EVALSHA`, `SCRIPT`), the `SCAN` family,
`PUBLISH`/`PUBSUB`, and the HotKV commands: tags
(`tag_set`/`tag_get`/`tag_members`/...), nanosecond TTLs
(`nexpire`/`nttl`/`nexpireat`), hot key tracking (`hotkeys_top`), the GCRA
rate limiter primitive (`gcra_limit`), the vector similarity index
(`vsim_*`), semantic and prompt caches (`scache_*`/`pcache_*`), the tool
call/negative/guardrail/reranker caches (`toolcache_*`/`ncache_*`/
`guard_*`/`rerank_*`), LLM usage stats and budgets (`llmstats_*`/
`llmbudget_*`), a versioned prompt registry (`prompt_*`), agent memory
(`amem_*`), embedding dedup (`embed_dedup_*`), a RAG pipeline (`rag_*`),
request rate limiting (`ratelimit_*`), time series (`ts_*`), a feature
store (`feature_*`, including `feature_setat()`) and observability
(`observe_*`/`alerts_*`). Each maps directly to its wire command; see the
docstrings for exact argument order and reply shape.

These have **no** typed method; send them with `client.execute()` (see
below), which returns the reply parsed but not reshaped:

- `CLIENT` other than `SETNAME`/`GETNAME`/`ID`: `LIST`, `INFO`, `KILL`,
  `PAUSE`, `UNPAUSE`, `UNBLOCK`, `REPLY`, `SETINFO`, `NO-EVICT`, `NO-TOUCH`,
  and the tracking commands `TRACKING`, `CACHING`, `GETREDIR`, `TRACKINGINFO`
- `ACL`, the `CLUSTER` administration commands, `CONFIG REWRITE`/`RESETSTAT`,
  `COMMAND`, `MEMORY`, `OBJECT`, `SLOWLOG`, `LATENCY`, `MODULE`, `DEBUG`,
  `SAVE`/`BGSAVE`, `MONITOR`, `REPLICAOF`, `FAILOVER`, `MIGRATE`
- the Bloom filter (`BF.*`), count-min sketch (`CMS.*`) and top-k (`TOPK.*`)
  commands
- the newer commands `HGETDEL`, `HGETEX`, `HSETEX`, `MSETEX`, `DELEX`,
  `HIMPORT`, `XCFGSET`, `SFLUSH`, `CLUSTERSCAN`, `TRIMSLOTS`, and also `SORT`,
  `LCS`, `LMPOP`/`BLMPOP`/`ZMPOP`/`BZMPOP`, `XSETID`, `GEORADIUS*`, `FCALL`/
  `FUNCTION`, `EVAL_RO`/`EVALSHA_RO`
- the enterprise management families `TENANT.*`, `IAM.*`, `AUDIT.*`,
  `ENCRYPTION.*`, `REPLICATION.*`

The enterprise engines need a license: without one the server answers
`NOLICENSE` and the integration tests skip themselves. The unit tests check the
exact wire arguments of the enterprise methods, and the integration tests run
them against a licensed server.

Floats are written as plain decimals (`1e-05` goes out as `0.00001`, never an
exponent), NaN is refused, and money amounts (`llmbudget_set()`'s cost limit,
`llmstats_price_set()`'s prices) are fixed point with at most 6 decimals.
`hotkv.format_float()` and `hotkv.format_usd()` are the functions that do it,
for building a score bound such as `"(" + hotkv.format_float(x)`.

```python
client.tag_set("doc:42", "draft", "internal")
client.tag_members("draft")  # every key tagged "draft"

client.gcra_limit("api:user:42", period_ms=60_000, burst=100)
```

Enterprise features need a license that includes them. A command whose feature
is not licensed raises `hotkv.NotLicensedError` (a subclass of
`hotkv.ResponseError`), which you can catch and handle distinctly:

```python
try:
    client.pcache_create("responses")
except hotkv.NotLicensedError:
    ...  # prompt caching is not included in this license
```

## AI caching

`hotkv.ai` provides two provider neutral helpers on top of the raw
commands (neither depends on any specific LLM or embedding SDK):

**`PromptCache`**: exact-match caching keyed by provider, model, messages
and generation parameters. `get_or_call()` only calls your LLM when there
is a cache miss:

```python
from hotkv.ai import PromptCache

cache = PromptCache(client, namespace="chat-responses", ttl=3600)

def call_llm():
    # however you call your provider, e.g. an OpenAI or Anthropic client
    return my_llm_client.chat(model="gpt-6-sol", messages=messages)

response, was_cached = cache.get_or_call(
    provider="openai",
    model="gpt-6-sol",
    messages=[{"role": "user", "content": "Summarize this in one line."}],
    fn=call_llm,             # only invoked on a miss
    stats_bucket="chat",     # optional: records hits/tokens via LLMSTATS.*
    tokens=(120, 40),
    cost_usd=0.0009,
)
if not was_cached:
    print("called the LLM")
```

**`SemanticCache`**: similarity based caching. You supply the embedding
vector (from whatever provider you use); it looks up the nearest cached
answer above a similarity threshold:

```python
from hotkv.ai import SemanticCache

cache = SemanticCache(client, namespace="faq", dims=1536, threshold=0.85)

def call_llm():
    return my_llm_client.chat(model="gpt-6-sol", messages=messages)

embedding = my_embedding_client.embed(user_question)
response, was_cached = cache.get_or_call(embedding, call_llm, query_text=user_question)
```

`hotkv.ai.AsyncPromptCache` and `hotkv.ai.AsyncSemanticCache` are the
asyncio equivalents, taking a `hotkv.asyncio.HotKV` client; their `fn` may
be a sync or an async callable.

**Recording usage so hits report savings**: every cache `set()` (and
`get_or_call()`, when `fn` returns `(response, usage)`) takes an optional
`hotkv.TokenUsage`, the original (uncached) call's token usage. A later hit
then attributes those tokens, and once the model is priced, their exact
cost, to the namespace's `usage_saved_*` stats:

```python
from hotkv import TokenUsage

def call_llm():
    response = my_llm_client.chat(model="claude-sonnet-5", messages=messages)
    usage = TokenUsage(
        model="claude-sonnet-5",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    return response.text, usage  # get_or_call() stores `usage` on a miss

response, was_cached = cache.get_or_call(
    provider="anthropic", model="claude-sonnet-5", messages=messages, fn=call_llm,
)
print(cache.stats())  # usage_saved_requests, usage_saved_input_tokens, ..., usage_saved_cost_usd
```

## LLM cost tracking

`LLMSTATS.*` and `LLMBUDGET.*` track token usage and cost independently of
caching, e.g. across every call your application makes to a given model:

```python
client.llmstats_bucket("chat")
client.llmstats_record(
    "chat", 100, 20, 0, 0, 120.5, model="claude-sonnet-5",
)  # cost_usd is ignored and priced by the server when `model` is given
print(client.llmstats_get("chat"))  # requests, cost_usd, cache_write_5m_tokens, ...

# Custom pricing for a model not in HotKV's built-in price table
client.llmstats_price_set(
    "my-fine-tuned-model", input_usd_per_mtok=2, output_usd_per_mtok=8,
)
print(client.llmstats_price_get("my-fine-tuned-model"))

client.llmbudget_set("team-alpha", token_limit=1_000_000, cost_limit_usd=50.0, window_seconds=86400)
allowed = client.llmbudget_consume(
    "team-alpha", usage=TokenUsage(model="claude-sonnet-5", input_tokens=100, output_tokens=20)
)
```

For OpenAI-style responses, whose `prompt_tokens` already includes cache
hits, pass `prompt_tokens - cached_tokens` as `llmstats_record()`'s
`input_tokens` (`cached_tokens` is always cache-READ tokens).

## Integrations

Optional integrations with popular AI frameworks live under
`hotkv.integrations` and require an extra (`pip install "hotkv-client[langchain]"`,
`"hotkv-client[llamaindex]"` or `"hotkv-client[litellm]"`); the base `hotkv-client` package never
depends on any of them, and importing one without its extra installed
raises a clear `ImportError` telling you what to install.

**LangChain** (`hotkv.integrations.langchain`, extra `langchain`):

```python
from langchain_core.globals import set_llm_cache
from hotkv.integrations.langchain import HotKVCache

set_llm_cache(HotKVCache(client, namespace="chat-responses", ttl=3600))
```

Both `HotKVCache` and `HotKVSemanticCache` record `hotkv.TokenUsage` with
each entry automatically whenever the cached `AIMessage` carries a model
name and LangChain's standard `usage_metadata`, so their savings show up in
`PCACHE.STATS`/`SCACHE.STATS`'s `usage_saved_*` fields with no extra code;
a plain text completion (no `usage_metadata`) is cached exactly as before.

`HotKVSemanticCache(client, embedding, threshold=0.86)` caches by
embedding similarity instead of exact text, one `SCACHE.*` namespace per
`llm_string` so different models never share answers;
`HotKVVectorStore(client, embedding, index_name="docs")` is a `VectorStore`
on `VSIM.*` (`add_texts`/`similarity_search`/`similarity_search_with_score`/
`delete`/`from_texts`); `HotKVChatMessageHistory(client, session_id="user-42")`
is a `BaseChatMessageHistory` backed by a HotKV list.

**LlamaIndex** (`hotkv.integrations.llama_index`, extra `llamaindex`):

```python
from hotkv.integrations.llama_index import HotKVVectorStore

vector_store = HotKVVectorStore(client, index_name="docs", dims=1536)
```

`HotKVKVStore(client)` is a `BaseKVStore` (for a `KVDocumentStore`/index
store) on HotKV hashes; `HotKVChatStore(client)` is a `BaseChatStore` for
`ChatMemoryBuffer` and friends.

**LiteLLM** (`hotkv.integrations.litellm`, extra `litellm`):

```python
import litellm
from hotkv.integrations.litellm import HotKVCache

litellm.cache = litellm.Cache()
litellm.cache.cache = HotKVCache(client, namespace="litellm-cache")
```

`HotKVSemanticCache(client, embedding_model="text-embedding-3-small")` adds
similarity based caching the same way. LiteLLM's own built-in
`litellm.Cache(type="redis", ...)` also works against HotKV unmodified for
exact caching, since HotKV speaks the RESP wire protocol; reach for
`hotkv.integrations.litellm` when you want semantic caching too, or want
caching to keep working on a server with no `PromptCache` license (see
below).

All three frameworks' exact-match caches (`HotKVCache`) use `PCACHE.*` and
transparently fall back to core `GET`/`SET` (a SHA-256 derived key, with
`EX` for the TTL) the first time the server answers `NOLICENSE`, so they
keep working on a Starter plan license or an unlicensed server, just
without `PCACHE.STATS` accounting.

## Error handling

Every exception derives from `hotkv.HotKVError`:

```python
import hotkv

try:
    client.set("key", "value")
    client.lpush("key", "x")  # key is a string, not a list
except hotkv.WrongTypeError as exc:
    print("wrong type:", exc)
except hotkv.NotLicensedError as exc:
    print("not licensed:", exc)
except hotkv.ResponseError as exc:
    print("server error", exc.code, exc)
except hotkv.ConnectionError:
    print("could not reach the server")
```

Key types: `ConnectionError`, `TimeoutError`, `AuthenticationError`,
`ProtocolError`, `ResponseError` (with `.code`, the leading error token,
and typed subclasses `WrongTypeError`, `NoAuthError`, `NotLicensedError`,
`NoScriptError`, `BusyError`, `MovedError`, `AskError`,
`ClusterDownError`, ...), `WatchError` and `ClusterError`. A missing key or
field is `None`, never an exception.

## Escape hatch

Any command without a typed method can be sent directly; arguments are
`str`, `bytes`, `int` or `float` (floats as above), and the reply is parsed
but not reshaped:

```python
client.execute("OBJECT", "ENCODING", "mykey")
client.execute("BF.ADD", "filter", "item")
client.execute("HGETEX", "h", "EX", 100, "FIELDS", 1, "field")
```

`client.command()` is the same method.

## Testing

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/unit            # no server needed
HOTKV_TEST_URL=hotkv://127.0.0.1:6379 python -m pytest tests/integration
```

Integration tests that need a license skip themselves with a clear reason
when the server replies `NOLICENSE`. Integration tests for
`hotkv.integrations.*` skip themselves the same way when the matching
framework extra is not installed.

One optional input extends the integration run:

- `HOTKV_REMOTE_HOST` (with `HOTKV_REMOTE_PORT`, `HOTKV_REMOTE_PASSWORD`,
  `HOTKV_REMOTE_CA`) runs `tests/integration/test_remote.py`, which uses the client
  as a customer would against a TLS and password protected instance over the
  network, with a random key prefix and its own cleanup; it touches nothing
  global. Pass the password only through the environment.

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Support

- Bugs and feature requests: [GitHub issues](https://github.com/hotkv-ltd/hotkv-python-client/issues)
- Questions and commercial support: support@hotkv.com
- Security vulnerabilities: see [SECURITY.md](SECURITY.md)

## License

Licensed under the Apache License, Version 2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Trademarks and affiliation

"HotKV" is a trademark of HotKV Ltd; the license does not grant rights to use it.
The HotKV server is a separate commercial product, and its enterprise commands need a HotKV license.

HotKV is an independent product of HotKV Ltd. It speaks the RESP protocol and implements
many Redis commands so that existing tools and client habits carry over, but it is not Redis,
Valkey, Dragonfly or KeyDB, and this SDK is built for and tested against HotKV. HotKV Ltd is not
affiliated with, endorsed by or sponsored by Redis Ltd., the Valkey project, DragonflyDB or KeyDB.
Redis is a registered trademark of Redis Ltd. Valkey, Dragonfly, KeyDB and all other product and
company names are trademarks of their respective owners; they are used here only to describe
protocol and command compatibility.
