import re
import sqlite3
from typing import Dict, List, Tuple, Optional

class SQLUtils:
    def __init__(self, path):
        self.path = path
        self.conn = self.create_connection()

    def create_connection(self):
        conn = sqlite3.connect(self.path)
        return conn

    def close_connection(self):
        if self.conn:
            self.conn.close()

    @staticmethod
    def execute_with_conn_check(func):
        def wrapper(self: "SQLUtils", *args, **kwargs):
            if not self.check_connection():
                self.conn = self.create_connection()
            return func(self, *args, **kwargs)
        return wrapper

    @execute_with_conn_check
    def execute_query(self, query, params=()):
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()
        return cursor.fetchall()

    def check_connection(self):
        try:
            self.conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def extract_query_from_response(self, response: str):
        if "```sql" in response and "```" in response[response.index("```sql") + len("```sql"):]:
            start = response.index("```sql") + len("```sql")
            end = response.index("```", start)
            return response[start:end].strip()
        return None

    # ---------- NEW: schema context for LLMs ----------
    @execute_with_conn_check
    def get_schema_context(self, include_views: bool = False) -> str:
        cur = self.conn.cursor()
        types = ("'table'", "'view'") if include_views else ("'table'",)
        cur.execute(
            f"""
            SELECT name, type, sql
            FROM sqlite_master
            WHERE type IN ({",".join(types)}) AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
            """
        )
        entries = cur.fetchall()  # [(name, type, sql), ...]

        lines: List[str] = []
        for name, obj_type, create_sql in entries:
            if obj_type == "view":
                # Views: summarize succinctly
                view_def = re.sub(r"\s+", " ", create_sql or "").strip()
                lines.append(f"VIEW {name}:")
                lines.append(f"- DEFINITION: {view_def}")
                lines.append("")  # spacer
                continue

            # ----- Columns & PK order -----
            cols_info = self._pragma_table_info(name)
            # PRAGMA table_info pk column gives order (0=not PK, 1..N = sequence)
            pk_cols_ordered = [c["name"] for c in sorted(cols_info, key=lambda r: r["pk"]) if c["pk"] > 0]

            lines.append(f"TABLE {name}:")
            lines.append("- COLUMNS:")

            for c in cols_info:
                parts = [c["type"] or ""]  # TYPE
                if c["notnull"]:
                    parts.append("NOT NULL")
                if c["pk"]:
                    parts.append(f"PK{c['pk'] if c['pk'] > 1 else ''}")  # PK or PK2/PK3...
                if c["dflt_value"] is not None:
                    # Keep raw default (SQLite returns SQL literal/string as stored)
                    parts.append(f"DEFAULT {c['dflt_value']}")
                # Column-level UNIQUE is represented via indexes/constraints; picked up below
                spec = " ".join(str(p) for p in parts if p)
                lines.append(f"  - {c['name']}: {spec}".rstrip())

            # ----- Primary Key (composite / explicit) -----
            # If composite or explicit PK in table SQL, show as a line
            if len(pk_cols_ordered) > 1:
                lines.append(f"- PRIMARY KEY: ({', '.join(pk_cols_ordered)})")
            elif len(pk_cols_ordered) == 1:
                # Single-column PK is already marked per-column; only show here if it appears as table-level clause (best-effort)
                if self._has_table_level_pk(create_sql):
                    lines.append(f"- PRIMARY KEY: ({pk_cols_ordered[0]})")

            # ----- Unique constraints via indexes -----
            unique_groups = self._unique_indexes(name)
            if unique_groups:
                uniques_fmt = ["(" + ", ".join(cols) + ")" for cols in unique_groups]
                lines.append(f"- UNIQUE: " + "; ".join(uniques_fmt))

            # ----- Foreign keys -----
            fks = self._foreign_keys(name)
            if fks:
                lines.append("- FOREIGN KEYS:")
                for fk in fks:
                    local = ", ".join(fk["from"])
                    ref = ", ".join(fk["to"])
                    clauses = []
                    if fk["on_update"]:
                        clauses.append(f"ON UPDATE {fk['on_update']}")
                    if fk["on_delete"]:
                        clauses.append(f"ON DELETE {fk['on_delete']}")
                    if fk["match"]:
                        clauses.append(f"MATCH {fk['match']}")
                    suffix = f" {' '.join(clauses)}" if clauses else ""
                    lines.append(f"  - ({local}) REFERENCES {fk['table']}({ref}){suffix}")

            # ----- Non-unique indexes (user-defined only) -----
            idxs = self._indexes(name)
            if idxs:
                lines.append("- INDEXES:")
                for idx in idxs:
                    if idx["unique"]:
                        continue  # uniques already reported above
                    if idx["name"].startswith("sqlite_autoindex_"):
                        continue
                    lines.append(f"  - {idx['name']}: ({', '.join(idx['columns'])})")

            # ----- CHECK constraints (best-effort) -----
            checks = self._extract_checks(create_sql or "")
            if checks:
                lines.append("- CHECKS:")
                for ch in checks:
                    lines.append(f"  - {ch}")

            lines.append("")  # spacer between tables

        return "\n".join(lines).rstrip()  # no trailing newline

    # ---------- helpers for schema extraction ----------
    def _pragma_table_info(self, table: str) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({self._ident(table)});")
        rows = cur.fetchall()
        # cid, name, type, notnull, dflt_value, pk
        return [
            {
                "cid": r[0],
                "name": r[1],
                "type": r[2],
                "notnull": bool(r[3]),
                "dflt_value": r[4],
                "pk": int(r[5]),
            }
            for r in rows
        ]

    def _foreign_keys(self, table: str) -> List[Dict]:
        """
        Returns grouped FKs where each entry may span multiple columns (id mapping).
        """
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA foreign_key_list({self._ident(table)});")
        rows = cur.fetchall()
        # (id, seq, table, from, to, on_update, on_delete, match)
        groups: Dict[int, Dict] = {}
        for r in rows:
            rid = int(r[0])
            seq = int(r[1])
            ref_table = r[2]
            from_col = r[3]
            to_col = r[4]
            on_update = (r[5] or "").upper() or None
            on_delete = (r[6] or "").upper() or None
            match = (r[7] or "").upper() or None
            g = groups.setdefault(
                rid,
                {"table": ref_table, "from": [], "to": [], "on_update": on_update, "on_delete": on_delete, "match": match},
            )
            # maintain order by seq
            g["from"].append(from_col)
            g["to"].append(to_col)
        return list(groups.values())

    def _unique_indexes(self, table: str) -> List[List[str]]:
        """
        Returns a list of unique column groups from UNIQUE indexes/constraints.
        """
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA index_list({self._ident(table)});")
        idx_rows = cur.fetchall()
        groups: List[List[str]] = []
        for r in idx_rows:
            idx_name = r[1]
            unique = bool(r[2])
            if not unique:
                continue
            if idx_name.startswith("sqlite_autoindex_"):  # internal
                continue
            # columns for this index
            cur.execute(f"PRAGMA index_info({self._ident(idx_name)});")
            cols = [row[2] for row in cur.fetchall()]  # seqno, cid, name
            if cols:
                groups.append(cols)
        return groups

    def _indexes(self, table: str) -> List[Dict]:
        """
        Returns user-defined indexes (both unique and non-unique) with their columns.
        """
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA index_list({self._ident(table)});")
        idx_rows = cur.fetchall()
        idxs: List[Dict] = []
        for r in idx_rows:
            idx_name = r[1]
            unique = bool(r[2])
            origin = r[3] if len(r) > 3 else None  # may be 'c', 'u', 'pk' or None depending on SQLite version
            if idx_name.startswith("sqlite_autoindex_"):
                continue
            cur.execute(f"PRAGMA index_info({self._ident(idx_name)});")
            cols = [row[2] for row in cur.fetchall()]
            idxs.append({"name": idx_name, "unique": unique, "origin": origin, "columns": cols})
        return idxs

    def _extract_checks(self, create_sql: str) -> List[str]:
        """
        Best-effort extraction of CHECK constraints from CREATE TABLE DDL.
        Handles both table-level and column-level CHECKs.
        """
        sql = " ".join(create_sql.split())  # normalize whitespace
        checks: List[str] = []

        # Table-level CHECK (...) patterns
        for m in re.finditer(r"\bCHECK\s*\((.*?)\)", sql, flags=re.IGNORECASE):
            expr = m.group(1).strip()
            if expr:
                checks.append(expr)

        # Column-level CHECK constraints (rough heuristic):
        # look for "<colname> <type> ... CHECK (...)" fragments between commas
        # This is intentionally conservative to avoid noisy output.
        column_defs = self._split_columns_from_create(sql)
        for coldef in column_defs:
            m = re.search(r"\bCHECK\s*\((.*?)\)", coldef, flags=re.IGNORECASE)
            if m:
                expr = m.group(1).strip()
                if expr and expr not in checks:
                    checks.append(expr)
        return checks

    def _split_columns_from_create(self, create_sql: str) -> List[str]:
        """
        Splits the CREATE TABLE (...) body into comma-separated items
        (columns and constraints), without naively breaking on commas inside parens.
        """
        m = re.search(r"\(\s*(.*)\)\s*(?:WITHOUT ROWID)?\s*;?\s*$", create_sql, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return []
        body = m.group(1)
        items = []
        buf = []
        depth = 0
        for ch in body:
            if ch == "(":
                depth += 1
                buf.append(ch)
            elif ch == ")":
                depth -= 1
                buf.append(ch)
            elif ch == "," and depth == 0:
                items.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        if buf:
            items.append("".join(buf).strip())
        # Return only items that look like column definitions (start with an identifier)
        return [it for it in items if re.match(r"^[`\"\[\]\w]", it)]

    def _has_table_level_pk(self, create_sql: Optional[str]) -> bool:
        if not create_sql:
            return False
        return bool(re.search(r"\bPRIMARY\s+KEY\s*\(", create_sql, flags=re.IGNORECASE))

    def _ident(self, name: str) -> str:
        # simple identifier wrapper for PRAGMA calls (no parameter binding available)
        return '"' + name.replace('"', '""') + '"'
