# Granforge demo backend

Granforge is an R&D playground that experiments with re‑implementing common database
operators in modern C++ and wiring them into Apache Arrow Acero plans.  The `granforge_demo`
binary is a CLI backend that reads Substrait plans produced by DuckDB, rewrites selected nodes
so they call the in‑tree operators, runs the plan with Arrow, and prints the results both as an
Arrow table and as JSON.

- Reference data set: https://doi.org/10.5281/zenodo.17803141
- Source layout: operators live in `include/granforge/dbops`, Arrow bridges live in
  `include/granforge/bridges/arrow`, and the CLI is in `src/examples/granforge_demo.cpp`.

## What the demo does

`granforge_demo` acts as an execution shim between DuckDB’s Substrait exporter and Arrow.  It:

1. Initializes Arrow Compute, spdlog, and registers custom ExecNode factories.
2. Reads a binary Substrait plan from standard input and a set of `--data TABLE=PATH` CLI flags
   that bind named tables in the plan to local Parquet files.
3. Deserializes the plan into an `arrow::acero::Declaration` tree and rewrites nodes like
   `scan`, `filter`, `hash_join`, `order_by`, `aggregate`, and `fetch` so they target the
   `chorys_*` factories implemented in this repo.
4. Executes the rewritten plan via `arrow::acero::DeclarationToTable`, producing the result table.
5. Logs the original and rewritten declarations for introspection and prints both a pretty Arrow
   table and a JSON payload that e2e tests can compare against DuckDB’s output.

The CLI intentionally stays small so the heavy lifting happens in the reusable dbops and bridge
layers rather than in ad‑hoc application code.

## dbops primitives

The `granforge::dbops` namespace provides composable, header‑only operators that implement the
data‑processing kernels.  Highlights:

- `parquet_scan_functor.hpp` – minimal Parquet reader that streams row groups into callbacks.
- `filter_mask_functor.hpp` & `span_comparison_functor.hpp` – apply predicate masks and evaluate
  vectorized comparisons that mirror DuckDB semantics.
- `group_by_functor.hpp` – COUNT aggregation over key columns using compact hash tables.
- `order_by_functor.hpp` – constructs ordering indices that can be applied to Arrow columns.
- `hash_join_functor.hpp` – integer equi‑join helper with build/probe phases split apart.
- `limit_functor.hpp` – maintains running offset/count windows that match LIMIT/OFFSET behavior.

Each functor deliberately exposes a narrow surface so it can be embedded into different runtimes
(Arrow, DuckDB extensions, tests) without taking a dependency on Arrow’s concrete classes.

## Arrow bridge layer

Files under `include/granforge/bridges/arrow` adapt the dbops functors into Arrow ExecNodes:

- `read.hpp` wraps the Parquet scan functor in an `ExecNode` that produces `ExecBatch` objects.
- `filter.hpp` redirects filter declarations to either re‑evaluated Arrow expressions or the
  dbops span comparison kernel, depending on the predicate.
- `group_by.hpp`, `order_by.hpp`, `hash_join.hpp`, and `limit.hpp` expose the corresponding dbops
  helpers to Acero, translating from Arrow buffers into the light‑weight functor interfaces.
- `register_ops.hpp` centralizes registration so the demo can idempotently enable every custom
  factory before accepting a plan.

This separation lets you exercise the dbops kernels inside Arrow, reuse them somewhere else, or
swap in new kernels without touching the demo binary.

## Workflow / control flow

1. **Input collection** – DuckDB (or another Substrait producer) writes a binary plan to stdout;
   you pipe it into `granforge_demo` while passing one or more `--data table=parquet.parq`
   arguments so named tables resolve to Parquet sources.
2. **Registration** – The demo calls the helpers in `register_ops.hpp` to make sure the global
   ExecFactory registry knows about every `chorys_*` node.  It also builds a small extension
   registry so Substrait functions such as `lt` map to Arrow compute kernels.
3. **Plan rewrite** – After `arrow::engine::DeserializePlan`, the tree of `acero::Declaration`
   nodes is walked by `RewritePlan`.  Each matching node swaps its `factory_name` to the custom
   implementation and (when needed) converts the options object to the wrapper type that understands
   dbops functors.
4. **Execution** – The demo feeds the rewritten declaration to
   `acero::DeclarationToTable`, which drives the ExecPlan.  Each custom node delegates to its
   dbops functor whenever it needs to evaluate predicates, build hash tables, or enforce limits.
5. **Result handling** – The resulting Table is streamed to stdout in human‑readable form, and a
   secondary JSON serialization (`JSON_RESULT:...`) keeps automated comparison runs simple.  Logs
   describing both plan variants are emitted via spdlog (path defaults to `granforge_backend.log`,
   override with `GRANFORGE_LOG_PATH`).

The control flow is intentionally linear: read plan → rewrite → execute → emit.  Most complexity
resides inside the reusable components, not the demo runner.

## Building and running

```bash
# Clone the repo and set up a build directory.
git clone git@github.com:JPietrzykTUD/granforge.git && cd granforge
# Run granforge_demo.sh (will build the project, download the reference dataset if needed, and run an example).
bash ./granforge_demo.sh
```

## Using the CMake package

Install the headers and CMake package config:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /opt/granforge
```

Consume from another CMake project:

```cmake
find_package(granforge CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE granforge::granforge)
```
