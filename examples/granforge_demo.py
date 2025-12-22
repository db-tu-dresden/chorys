#!/usr/bin/env python3
"""
GranForge Substrait Demo CLI.

This module wires DuckDB, Substrait, and our Arrow backend together.  It can be
imported by Sphinx' autodoc to expose the functions/classes documented here or
executed as a script.  The CLI accepts structured arguments to ingest data,
produce Substrait plans, and hand the plans to a backend executable.
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings  # currently unused, but kept
MAX_HISTORY = 100

class LimitedHistory(InMemoryHistory):
    """History that keeps only the latest MAX_HISTORY entries."""

    def append_string(self, string: str) -> None:
        """
        Add a prompt entry but truncate the oldest entries once the cap is exceeded.

        Parameters
        ----------
        string:
            Raw text to store in the in-memory buffer.
        """
        super().append_string(string)
        # Trim history to the last MAX_HISTORY items
        if len(self._loaded_strings) > MAX_HISTORY:
            overflow = len(self._loaded_strings) - MAX_HISTORY
            del self._loaded_strings[0:overflow]


import argparse
import subprocess
import time
from pathlib import Path
import glob
import sys
import os
import tempfile
import duckdb
import re
import logging
import pydoc
import json

from google.protobuf import json_format
from substrait import proto  # <-- this provides proto.Plan, proto.ReadRel, etc.


def setup_logging(verbose: bool = False, quiet: bool = False):
    """
    Setup global logging with reasonable defaults:
    - INFO to stderr
    - DEBUG when --verbose
    - suppress most logs when --quiet
    """
    log_level = logging.INFO
    if verbose:
        log_level = logging.DEBUG
    if quiet:
        log_level = logging.WARNING

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,    # override any existing configuration
    )

    # Reduce noise from libraries
    logging.getLogger("duckdb").setLevel(logging.WARNING)


def init_duckdb(db_path: str | None,
                schema: dict[str, str] | None,
                tables: dict[str, list[str]]) -> duckdb.DuckDBPyConnection:
    """
    Create a DuckDB connection and populate it with user-provided tables.

    Parameters
    ----------
    db_path:
        Path to a DuckDB database or ``None`` for in-memory usage.
    schema:
        Optional mapping ``table_name -> CREATE TABLE statement`` describing the
        destination tables.  When provided, loaded data is inserted instead of
        creating new tables.
    tables:
        Mapping ``table_name -> [paths...]`` containing the physical data files
        to ingest (Parquet/CSV/TSV).  Paths are expanded/globbed by ``parse_table_args``.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Connection with tables loaded and statistics analyzed.
    """

    db = ":memory:" if db_path is None else db_path
    con = duckdb.connect(db, config={
        "allow_unsigned_extensions": "true",
        "enable_external_access": "true",
    })
    con.execute("LOAD 'substrait';")

    if schema:
        for table_name, create_stmt in schema.items():
            con.execute(create_stmt)
        stmt_prefix = "INSERT INTO"
    else:
        stmt_prefix = "CREATE TABLE"

    def ingest_prefix_stmt(table_name: str) -> str:
        if schema:
            return f"INSERT INTO {table_name} "
        else:
            return f"CREATE TABLE {table_name} AS "

    for table_name, paths in tables.items():
        for path in paths:
            path = str(Path(path).resolve())
            suffix = Path(path).suffix.lower()
            logging.info(f"Loading data for table '{table_name}' from {path}...")

            if suffix == ".parquet":
                # Materialize as real table so we can ANALYZE later
                con.execute(
                    f"{ingest_prefix_stmt(table_name)} SELECT * FROM read_parquet(?)",
                    [path],
                )
            elif suffix in (".csv", ".tsv"):
                con.execute(
                    f"""
                    {ingest_prefix_stmt(table_name)} SELECT * FROM read_csv_auto(?, header=TRUE)
                    """,
                    [path],
                )
            else:
                raise ValueError(f"Unsupported file type for {table_name}: {path}")

    if tables:
        con.execute("ANALYZE;")

    return con


def load_schema_from_file(schema_path: str) -> dict[str, str]:
    """
    Parse a schema definition file containing ``CREATE TABLE`` statements.

    Parameters
    ----------
    schema_path:
        Path to the schema file, one statement per line.

    Returns
    -------
    dict[str, str]
        Mapping of table name to the full ``CREATE TABLE`` statement.

    Raises
    ------
    ValueError
        If the file contains invalid lines or duplicates.
    """
    tables: dict[str, str] = {}

    create_table_regex = re.compile(
        r"^CREATE\s+TABLE\s+\"?(?P<table_name>\w+)",
        re.IGNORECASE
    )

    for line in Path(schema_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        match = create_table_regex.match(stripped)
        if match:
            name = match.group("table_name")
            if name not in tables:
                tables[name] = stripped
            else:
                raise ValueError(f"Duplicate table definition in schema: {name}")
        else:
            raise ValueError(f"Invalid schema line (expected CREATE TABLE): {line}")

    return tables


def parse_table_args(table_args: list[str]) -> dict[str, list[str]]:
    """
    Parse --table inputs with rules:

        --table orders=/data/orders.parquet
        --table /data/lineitem_*.parquet
        --table lineitem=/data/*.parquet

    Output:
        { table_name: [file1, file2, ...] }
    """
    tables: dict[str, list[str]] = {}

    for t in table_args:
        if "=" in t:
            name, pattern = t.split("=", 1)
            name = name.strip()
            pattern = pattern.strip()
        else:
            # No explicit name → derive from filename
            pattern = t.strip()
            example_path = next(iter(glob.glob(pattern)), None)
            if example_path is None:
                continue
            name = None

        expanded = sorted(glob.glob(pattern))

        if not expanded:
            continue

        if name is None:
            for path in expanded:
                derived_name = Path(path).stem
                tables.setdefault(derived_name, []).append(path)
        else:
            tables.setdefault(name, []).extend(expanded)

    return tables


def select_tables_for_sql(sql: str,
                          available: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Return a subset of ``available`` containing tables referenced in ``sql``.
    Detection is best-effort and falls back to ``{}`` when no matches were found.
    """
    if not sql or not available:
        return {}

    normalized_sql = re.sub(r'[`\[\]\"]', " ", sql.lower())
    selected: dict[str, list[str]] = {}
    for table_name, paths in available.items():
        pattern = rf'\b{re.escape(table_name.lower())}\b'
        if re.search(pattern, normalized_sql):
            selected[table_name] = paths
    return selected


def fetch_all_results(cur: duckdb.DuckDBPyConnection, chunk_size=1000):
    """
    Consume a DuckDB cursor and return strings for printing.

    Parameters
    ----------
    cur:
        DuckDB cursor with a pending result set.
    chunk_size:
        Number of rows fetched per iteration.

    Returns
    -------
    list[str]
        Tab separated lines for each row.
    """
    rows = []
    while True:
        chunk = cur.fetchmany(chunk_size)
        if not chunk:
            break
        for row in chunk:
            rows.append("\t".join(map(str, row)))
    return rows


def format_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "(no rows)"
    columns = list(rows[0].keys())
    widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows))
              for col in columns}
    header = " | ".join(f"{col:{widths[col]}}" for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    lines = [header, separator]
    for row in rows:
        lines.append(" | ".join(f"{str(row.get(col, '')):{widths[col]}}" for col in columns))
    return "\n".join(lines)


def strip_read_projection_in_place(obj):
    """
    Recursively walk the Substrait JSON structure and drop
    'projection' fields inside 'read' relations.

    This is a workaround for Arrow's 'NotImplemented: substrait::ReadRel::projection'.
    """
    if isinstance(obj, dict):
        if "read" in obj and isinstance(obj["read"], dict):
            read_rel = obj["read"]
            projection = read_rel.pop("projection", None)
            if projection is not None:
                logging.info("Stripping ReadRel.projection")
                struct_items = (
                    projection.get("select", {}).get("structItems", [])
                )
                expressions = []
                for item in struct_items:
                    field_idx = item.get("field")
                    if field_idx is None:
                        field_idx = 0
                    expressions.append(
                        {
                            "selection": {
                                "directReference": {
                                    "structField": {"field": field_idx}
                                },
                                "rootReference": {},
                            }
                        }
                    )
                base_schema = read_rel.get("baseSchema", {})
                base_names = base_schema.get("names")
                if base_names and isinstance(base_names, list):
                    num_input_fields = len(base_names)
                else:
                    struct_info = base_schema.get("struct", {})
                    num_input_fields = len(struct_info.get("types", []))

                # Arrow ProjectRels append evaluated expressions to the original
                # input columns unless emit/outputMapping is provided.  DuckDB's
                # ReadRel.projection returns only the selected columns, so we emit
                # only the appended expressions to mirror those semantics.
                new_input = {"read": read_rel}
                project_rel = {
                    "input": new_input,
                    "expressions": expressions,
                }
                if expressions:
                    projected_indices = list(
                        range(num_input_fields, num_input_fields + len(expressions))
                    )
                    project_rel["common"] = {
                        "emit": {"outputMapping": projected_indices}
                    }

                obj.clear()
                obj["project"] = project_rel
                strip_read_projection_in_place(obj)
                return
        for v in obj.values():
            strip_read_projection_in_place(v)
    elif isinstance(obj, list):
        for item in obj:
            strip_read_projection_in_place(item)


def normalize_aggregate_phases_in_place(obj):
    """
    DuckDB may omit aggregation phases (or emit AGGREGATION_PHASE_UNSPECIFIED),
    but Arrow's Substrait consumer currently expects INITIAL_TO_RESULT.
    Walk the plan JSON and rewrite missing/unspecified phases accordingly.
    """
    if isinstance(obj, dict):
        if "aggregate" in obj and isinstance(obj["aggregate"], dict):
            measures = obj["aggregate"].get("measures", [])
            for measure in measures:
                if not isinstance(measure, dict):
                    continue

                agg = measure.get("measure")
                if not isinstance(agg, dict):
                    continue

                phase = agg.get("phase")
                if not phase or phase == "AGGREGATION_PHASE_UNSPECIFIED":
                    agg["phase"] = "AGGREGATION_PHASE_INITIAL_TO_RESULT"
        for v in obj.values():
            normalize_aggregate_phases_in_place(v)
    elif isinstance(obj, list):
        for item in obj:
            normalize_aggregate_phases_in_place(item)


def lift_read_filters_in_place(obj):
    """
    Arrow's ExecuteSerializedPlan expects filters to be represented as
    FilterRel nodes. DuckDB encodes WHERE clauses inside ReadRel.filter.
    This helper rewrites those reads into FilterRels so downstream engines
    can consume the plan without custom ReadRel handling.
    """
    if isinstance(obj, dict):
        # Only rewrite when this dict is exactly a relation with a single "read" key
        if set(obj.keys()) == {"read"} and isinstance(obj["read"], dict):
            read_rel = obj["read"]
            filter_expr = read_rel.pop("filter", None)
            if filter_expr is not None:
                new_input = {"read": read_rel}
                obj.clear()
                obj["filter"] = {
                    "input": new_input,
                    "condition": filter_expr,
                }
                # Continue rewriting within the new filter relation
                lift_read_filters_in_place(obj)
                return

        for v in obj.values():
            lift_read_filters_in_place(v)
    elif isinstance(obj, list):
        for item in obj:
            lift_read_filters_in_place(item)


def get_plan_for_statement(con: duckdb.DuckDBPyConnection, sql: str) -> bytes:
    """
    Generate a *binary* Substrait plan for `sql`, with a JSON rewrite to
    strip ReadRel.projection so Arrow's Substrait consumer can handle it.

    Steps:
    1) Get JSON Substrait plan from DuckDB (get_substrait_json).
    2) Verify that from_substrait_json(plan_json) == direct SQL results.
    3) Strip ReadRel.projection from the JSON.
    4) Convert the modified JSON into a binary substrait::Plan (protobuf bytes).
    5) Return those bytes (to be piped into the Arrow backend).
    """

    # 1) Get JSON plan from DuckDB
    plan_json = con.execute(
        "SELECT * FROM get_substrait_json(?);",
        [sql],
    ).fetchone()[0]
    Path("/tmp/plan_raw.json").write_text(plan_json, encoding="utf-8")

    # DuckDB returns TEXT; normalize to str explicitly (just in case)
    if isinstance(plan_json, memoryview):
        plan_json = plan_json.tobytes().decode("utf-8")
    else:
        plan_json = str(plan_json)

    # 2) Verify using from_substrait_json before we rewrite anything
    try:
        cur = fetch_all_results(
            con.execute("SELECT * FROM from_substrait_json(?);", [plan_json])
        )
        base = fetch_all_results(con.execute(sql))
    except Exception as e:
        logging.warning(
            "Failed to execute from_substrait_json for verification: %s",
            e,
            exc_info=True,
        )
        # If verification fails, we still return a rewritten plan; just skip the check
        cur = base = []

    if cur and base:
        if len(cur) != len(base):
            raise ValueError(
                f"Result row count mismatch between original SQL ({len(base)}) "
                f"and from_substrait_json ({len(cur)})"
            )

        for i, line in enumerate(cur):
            cur_line = line.strip()
            base_line = base[i].strip()
            if cur_line != base_line:
                raise ValueError(
                    "Result row mismatch at line {idx}:\n"
                    "  from_substrait_json: {cur}\n"
                    "  original SQL:       {base}".format(
                        idx=i, cur=cur_line, base=base_line
                    )
                )

        logging.info("from_substrait_json results match original SQL results")

    # 3) Strip ReadRel.projection from JSON to work around Arrow's limitation
    plan_dict = json.loads(plan_json)
    strip_read_projection_in_place(plan_dict)
    Path("/tmp/plan_rewritten.json").write_text(
        json.dumps(plan_dict), encoding="utf-8"
    )
    lift_read_filters_in_place(plan_dict)
    normalize_aggregate_phases_in_place(plan_dict)

    # 4) Convert modified JSON dict to substrait::Plan protobuf
    plan = proto.Plan()
    # json_format.ParseDict fills the Plan from a Python dict
    json_format.ParseDict(plan_dict, plan)

    # 5) Serialize to binary and return
    return plan.SerializeToString()


def run_query(sql: str,
              schema_tables: dict[str, str] | None,
              table_paths: dict[str, list[str]],
              backend_cmd: list[str] | None,
              chunk_size: int = 1000):
    """
    Execute `sql` against DuckDB and optionally a backend, compare results, and report timings.

    This spins up a fresh DuckDB connection, ingests the provided tables, executes
    the SQL directly and (when configured) via the backend, and asserts that both
    paths return identical textual results.  Execution times are logged for the
    paths that ran.
    """
    log = logging.getLogger(__name__)

    inferred_tables = select_tables_for_sql(sql, table_paths)
    if inferred_tables:
        tables_for_query = inferred_tables
    else:
        tables_for_query = table_paths
        if table_paths:
            log.debug("Falling back to loading all configured tables for query")

    schema_subset = None if not schema_tables else {
        name: stmt for name, stmt in schema_tables.items()
        if name in tables_for_query
    }
    if schema_tables and not schema_subset and not inferred_tables:
        schema_subset = schema_tables

    start_duck = time.perf_counter()
    duckdb_con = init_duckdb(db_path=":memory:",
                             schema=schema_subset,
                             tables=tables_for_query)
    duckdb_con.execute(sql)
    duck_tuples = duckdb_con.fetchall()
    columns = [col[0] for col in duckdb_con.description]
    duck_rows = [
        {columns[idx]: row[idx] for idx in range(len(columns))}
        for row in duck_tuples
    ]
    duck_time = time.perf_counter() - start_duck
    json_rows = json.dumps(duck_rows)

    if not backend_cmd:
        print("=== DuckDB result ===")
        print(format_table(duck_rows))
        print(f"[timing] duckdb={duck_time:.3f}s")
        return

    plan_bytes = get_plan_for_statement(duckdb_con, sql)

    fd, temp_path = tempfile.mkstemp(prefix="granforge_backend_", suffix=".log")
    os.close(fd)
    log_path = Path(temp_path)
    backend_env = os.environ.copy()
    backend_env["GRANFORGE_LOG_PATH"] = str(log_path)

    try:
        start_backend = time.perf_counter()
        proc = subprocess.run(backend_cmd,
                              input=plan_bytes,
                              capture_output=True,
                              env=backend_env)
        backend_time = time.perf_counter() - start_backend

        backend_log_text = ""
        if log_path.exists():
            backend_log_text = log_path.read_text(encoding="utf-8", errors="replace").strip()

        backend_stdout = proc.stdout.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            log.error("Backend failed with code %s:\n%s",
                      proc.returncode,
                      proc.stderr.decode("utf-8", errors="replace"))
            if backend_log_text:
                log.error("Backend log output:\n%s", backend_log_text)
            raise RuntimeError(f"Backend execution failed (exit code {proc.returncode})")

        backend_table_text = []
        backend_json = None
        for line in backend_stdout.splitlines():
            if line.startswith("JSON_RESULT:"):
                backend_json = line.split("JSON_RESULT:", 1)[1].strip()
            else:
                backend_table_text.append(line)

        if backend_json is None:
            log.error("Backend output lacked JSON_RESULT marker:\n%s", backend_stdout)
            raise AssertionError("Backend did not emit JSON_RESULT for verification")

        backend_rows = json.loads(backend_json)

        if duck_rows != backend_rows:
            log.error("DuckDB vs backend mismatch:\nDuckDB JSON:\n%s\nBackend JSON:\n%s",
                      json_rows, backend_json)
            print("Duck sample:", duck_rows[:5])
            print("Backend sample:", backend_rows[:5])
            raise AssertionError("Mismatch between DuckDB and backend results")

        print("=== Backend log ===")
        if backend_log_text:
            print(backend_log_text)
        else:
            print("(log empty)")
        print("=== End backend log ===")

        print("=== DuckDB result ===")
        print(format_table(duck_rows))
        print("=== Backend result ===")
        print(format_table(backend_rows))

        print(f"[timing] duckdb={duck_time:.3f}s backend={backend_time:.3f}s")
    finally:
        # Keeping backend log for debugging
        pass


def main():
    """
    CLI entry point.  Parses arguments, loads data into DuckDB,
    and either executes SQL directly or generates Substrait plans
    and forwards them to the backend.
    """

    parser = argparse.ArgumentParser(
        description="GranForge Demo with DuckDB and Prompt Toolkit"
    )
    parser.add_argument(
        "--db",
        default=":memory:",
        help="Path to DuckDB database file (default: in-memory)",
    )
    parser.add_argument(
        "--schema-file",
        default=None,
        help="Path to schema file defining tables (format: table_name=path_to_file, default: None, will get schema from data files)",
    )
    parser.add_argument(
        "--data",
        action="append",
        nargs="+",
        default=[],
        help="Table definitions (supports NAME=PATH or PATH with wildcards).",
    )
    parser.add_argument(
        "--sql-file",
        default=None,
        help="Path to SQL file to execute "
    )
    parser.add_argument(
        "--backend",
        help="Backend to use",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("-q", "--quiet",   action="store_true", help="Reduce log output")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)
    log = logging.getLogger(__name__)

    schema_tables = (
        load_schema_from_file(args.schema_file) if args.schema_file else None
    )

    table_args_tables = parse_table_args([item for group in args.data for item in group])

    # delete all schemas if there is no counterpart in table_args_tables
    if schema_tables:
        for table_name in list(schema_tables.keys()):
            if table_name not in table_args_tables:
                del schema_tables[table_name]

    if not table_args_tables:
        parser.error("At least one --data table=path is required")

    data_args: list[str] = []
    for table_name, paths in table_args_tables.items():
        if not paths:
            continue
        if len(paths) > 1:
            log.warning(
                "Multiple files provided for table '%s'; using the first path (%s)",
                table_name,
                paths[0],
            )
        data_args.extend(["--data", f"{table_name}={paths[0]}"])

    backend_cmd = [args.backend, *data_args] if args.backend else None
    if not backend_cmd:
        log.info("No backend provided; executing queries with DuckDB only")

    if args.sql_file:
        raw_sql_lines = Path(args.sql_file).read_text(encoding="utf-8").splitlines()
        sql_lines = [line for line in raw_sql_lines if not line.strip().startswith("--")]
        sql = "\n".join(sql_lines)
        for statement in sql.split(";"):
            statement = statement.strip()
            if not statement:
                continue
            log.info(f"Processing SQL statement:{statement}")
            run_query(statement, schema_tables, table_args_tables, backend_cmd)

        return
    else:
        log.info("Entering interactive SQL prompt (type Ctrl-D to exit)...")
        history = LimitedHistory()
        session = PromptSession(history=history)
        while True:
            try:
                line = session.prompt("Come forth, curious soul, cast thy SELECT> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            line = line.strip()
            if not line:
                continue

            run_query(line, schema_tables, table_args_tables, backend_cmd)


if __name__ == "__main__":
    main()
