# Changelog

All notable changes to lautpy are documented here. Versions follow the semver-ish `MAJOR.MINOR.PATCH` scheme while in 0.0.x.

## [Unreleased]

### Added
- `lautpy.agent.harness`: lightweight agent-harness pieces (DeepAgents-style) —
  `Harness` guards (tool-result truncation, repeat-call detection, approval
  gate, char budget with forced final answer, history compaction with optional
  summarizer) plus `make_todo_tool` (planning), `fs_tools` (sandboxed file
  tools for context offloading) and `make_subagent_tool` (agent-as-tool with
  isolated context); composable, all off by default
- `lautpy.ml`: machine-learning glue layer — `set_seed`, `best_threshold` /
  `ks_stat` / `psi` / `report_lite` (numpy-only metric gaps), quantile/uniform
  binning with WOE/IV + `woe_transform`, `xbenchmark` (parallel multi-model
  comparison), `xsplit` (stratified train/val/test) / `xy`, and
  `model_dump` / `model_load` with metadata; `ml` extra now includes pandas
- Agent engine upgrades: `run_agent(response_model=PydanticModel)` for
  validated structured output (invalid JSON is fed back for retry) and
  `run_agent_stream` for incremental streaming with in-band tool execution
- `lautpy.llm`: public `resolve_credentials()` / `resolve_model()` (were
  private, cross-module used); shared HTTP session with connection reuse via
  `apis.get_shared_session()` (default for `request()`)
- CI: mypy type check (17 files, 0 issues) alongside ruff
### Changed
- `uv.lock` removed from version control (library best practice) — native OpenAI
  tool-calling loop (`run_agent`, no langchain needed, Python 3.8+) and
  LangChain 1.x `create_agent` wrapper (`build_agent`, Python 3.10+, via the
  new `agent` extra); one `@tool` definition feeds both engines
- `docs/architecture.md`, `docs/usage.md`, this changelog

### Changed
- **Zero mandatory dependencies**: `requests` moved to the new `http` extra;
  `lautpy.apis` / `lautpy.notice` are lazily mounted (PEP 562) so
  `import lautpy` loads no third-party packages
- `ratelimit` waits via `threading.Condition` (lock released while waiting —
  parallel waiters no longer serialize behind a sleeping holder)
- `wecom`: `mentioned_mobile_list` preserved across message chunks (regression
  covered); `xgroup` accepts set/dict via explicit sliceable-type check;
  `BloomFilter.add` hashes once per add (was twice); `ABTest` docs state the
  half-open ranger semantics; `xAsyncio` raises a guided error inside a
  running event loop; `xsse_parser_` name matches the eager-suffix convention
  (`xsse_parser` kept as alias); ruff rule groups SIM/C4/PERF enabled
- **Python floor raised to >= 3.10**; typing modernized package-wide (`Optional[X]` → `X | None`, `List[X]` → `list[X]`, collections.abc imports); 3.7 importlib_metadata fallback removed; CI matrix now 3.10–3.13; ruff target py310 with UP rules enabled
- Unified logger fallback in a single internal module (`_internal.py`)
- `xThreadPoolExecutor`/`xProcessPoolExecutor` share one implementation
- `xsse_parser` logs JSON decode errors through the package logger instead of raw `print`
- `notice`: long messages are now split by UTF-8 byte length (CJK-safe), never mid-character; empty title no longer leaves a `****` markdown artifact
- Version number is now sourced solely from `.data/VERSION` (setuptools dynamic version); `release.sh` no longer patches pyproject.toml
- Added CI test workflow (Python 3.8–3.13 matrix + ruff) alongside the publish workflow
- Docstring pass across all modules: unified Google-style Args/Returns/Example (Chinese), accurate `__doc__` on plain Pipe instances, stale template headers removed; ruff configured and clean

## [0.0.6.0] — metadata release

- `requires-python >= 3.8` (was a false `>= 3.7`); added 3.13 classifier
- Optional dependencies exposed as extras: `retry` / `llm` / `yaml` / `ml` / `numpy` / `all` / `dev`
- `numpy` moved from a hard dependency to an extra (the code already treats it as optional)
- PEP 639 SPDX license metadata, `py.typed` marker, keywords, Development Status / Typing classifiers
- Author metadata unified; third-party references pruned from source docstrings

## [0.0.5.0]

- `decorators.py`: `retrying` (tenacity wrapper), `timeout`, `background_task`, `ratelimit`, `singleton`, `synchronized`, `tryer`
- `notice.py`: WeCom/Feishu webhook senders (env-var URLs, byte-safe chunking came later)
- `llm.py`: OpenAI-compatible client factory resolving `<SVC>_API_KEY` / `<SVC>_BASE_URL`
- `paths`: `pkl_dump`/`pkl_load`; `apis.tools`: `download`, `is_open`

## [0.0.4.0]

- `xUniquePlus` (pickle-fallback dedup), `xBloomFilter` (stdlib Bloom filter), `xHashBins` (stable hash bucketing) — all dependency-free rewrites
- `lautpy.apis` layer: `get_api_key` (env-only keys), `request` (timeout + retry), keyless tools (`shorten_url`, `data2qrcodeurl`), NiuTrans translate via `NIUTRANS_API_KEY`
- Guard test: CI fails if a hardcoded key ever lands in `apis/` source

## [0.0.3.0]

- `pipe.py`: eager variants (`xmap_`/`xfilter_`/`xchain_`/`xdrop`/`xdrop_`), dict helpers (`xchain_dict`/`xDictValues`/`xDictRemove`/`xgetitem`), `xCounterUpdate`, `xstack`, `xAsyncio`, `xnext`
- New modules: `dates`, `paths`, `hashing` (md5/murmurhash/ABTest)
- `__all__` whitelist to stop namespace pollution
- pytest suite introduced

## [0.0.2.4]

- Fixed crash on `import lautpy` (version detection, stale template header)
