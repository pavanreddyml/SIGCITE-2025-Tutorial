from typing import Literal, Optional, Dict, Any, List
import textwrap
import json

from ..llm.ollama_client import OllamaClient
from ..sql.sql_utils import SQLUtils


class SQLAgent:
    def __init__(self, ollama_client: OllamaClient, sql_utils: SQLUtils, privileges: str = "admin", dialect: str = "sqlite"):
        self.ollama_client = ollama_client
        self.sql_utils = sql_utils

        if privileges not in ["read", "write", "admin"]:
            raise ValueError("privileges must be 'read', 'write', or 'admin'")
        self.privileges = privileges
        self.dialect = dialect

        schema_context = sql_utils.get_schema_context()

        self.planner_prompt = textwrap.dedent(f"""
        You are a schema-aware SQL planner. Read the user's request and output a STRICT JSON object (no prose) that
        identifies the minimal set of tables/columns, relationships, filters, aggregates, ordering, and limits needed.

        SCHEMA (canonical; use exact identifiers):
        {schema_context}

        Rules:
        - Output ONLY JSON. No comments, no markdown.
        - Use exact table and column names as in SCHEMA.
        - Prefer single table when possible; include joins only when required.
        - Keep it minimal and executable.
        - action ∈ ["SELECT","INSERT","UPDATE","DELETE"].
        - Where applicable, include "date_columns" you touch; note SQLite date literal format 'YYYY-MM-DD'.
        - If ambiguous, choose reasonable defaults (ORDER BY name asc when selecting names, LIMIT 50).
        - If a constraint exists (PK, FK, CHECK), reflect it in "constraints_considered".
        - NEVER invent tables or columns.

        JSON schema:
        {{
          "action": "SELECT" | "INSERT" | "UPDATE" | "DELETE",
          "tables": ["Models", "Brands", ...],
          "columns": ["Models.model_name", "Brands.brand_name", ...],    # minimally needed projection/targets
          "joins": [{{"left":"Models.brand_id","right":"Brands.brand_id","type":"INNER"}}],
          "filters": ["LOWER(Brands.brand_name) LIKE LOWER('%ferrari%')", "Customers.gender = 'Male'", ...],
          "aggregations": ["COUNT(*) AS cnt", "AVG(Models.model_base_price) AS avg_price"],
          "group_by": ["Brands.brand_name", ...],
          "order_by": ["Brands.brand_name ASC", "cnt DESC"],
          "limit": 50,
          "date_columns": ["Car_Parts.manufacture_start_date"],
          "constraints_considered": ["Customers.gender ∈ {{'Male','Female'}}", "Dealer_Brand FK to Brands/Dealers"],
          "notes": ["why joins needed in one short phrase"]
        }}

        Output the JSON for this user request:
        """)

        # --- MAIN SQL GENERATION PROMPT (improved & schema-grounded) ---
        # This prompt is further augmented at runtime with "Context Hints" from the planner.
        self.system_prompt_base = textwrap.dedent(f"""
        ROLE:
        You translate user requests into a single, correct SQL statement. You have full privileges.

        SCHEMA (authoritative; use exact identifiers and semantics):
        {schema_context}

        SCOPE & PRIORITY:
        - Primary focus: Brands, Models, Customers and their relationships to Dealers, Car_Vins, Car_Options, Car_Parts, Manufacture_Plant.
        - Prefer single-table queries when possible. Join only when the request requires related data (e.g., "Ferrari models" -> Models ⋈ Brands).

        DIALECT:
        - Target SQL dialect: {{dialect}} (assume SQLite unless told otherwise).
        - Identifiers: use double quotes only when necessary (reserved words / mixed case); strings use single quotes.
        - Dates: 'YYYY-MM-DD'. Use SQLite date/time functions when needed.

        RELATIONAL MAP (use these canonical join paths; do NOT invent):
        - Models.brand_id → Brands.brand_id
        - Dealer_Brand.dealer_id → Dealers.dealer_id; Dealer_Brand.brand_id → Brands.brand_id
        - Car_Options.model_id → Models.model_id
        - Car_Options.engine_id / transmission_id / chassis_id / premium_sound_id → Car_Parts.part_id
        - Car_Vins.model_id → Models.model_id; Car_Vins.option_set_id → Car_Options.option_set_id; Car_Vins.manufactured_plant_id → Manufacture_Plant.manufacture_plant_id
        - Customer_Ownership.customer_id → Customers.customer_id; Customer_Ownership.vin → Car_Vins.vin; Customer_Ownership.dealer_id → Dealers.dealer_id
        - Car_Parts.manufacture_plant_id → Manufacture_Plant.manufacture_plant_id

        CONSTRAINTS (respect these in DML/filters):
        - CHECKs:
          * Customers.gender ∈ ("Male","Female")
          * Manufacture_Plant.plant_type ∈ ("Assembly","Parts")
          * Manufacture_Plant.company_owned ∈ (0,1)
          * Car_Parts.part_recall ∈ (0,1)
        - PK/UK/FK integrity must be maintained on INSERT/UPDATE/DELETE.

        HEURISTICS:
        - Default to SELECT; use INSERT/UPDATE/DELETE only if the user asks to add/change/remove/drop.
        - Fuzzy search: LOWER(column) LIKE LOWER('%term%').
        - Aggregates: COUNT/SUM/AVG with proper GROUP BY.
        - Ambiguity: pick reasonable defaults (ORDER BY name asc if a name is selected; LIMIT 50 for large outputs).
        - Prefer EXISTS over DISTINCT when checking membership; avoid unnecessary subqueries.
        - For multi-step logic, use CTEs (WITH ...) and return ONE final statement.

        DML SAFETY (user allows destructive ops; still be precise):
        - Always include WHERE clauses for UPDATE/DELETE unless the user explicitly wants a full-table operation.
        - For INSERTs, provide explicit column lists and values of correct types; satisfy FKs.
        - For DELETEs that touch children first (if needed), sequence via CTEs if supported, or rely on constraints (no ON DELETE CASCADE specified).

        OUTPUT CONTRACT (STRICT):
        - Output EXACTLY one fenced SQL code block using ```sql ... ``` and nothing else. No commentary.
        - If multiple CTE steps are needed, compose and return ONE final statement.
        - Use only tables/columns from SCHEMA. Never hallucinate names.

        EXAMPLES (patterns to emulate; adapt to request):
        -- 1) List Ferrari models
        SELECT m."model_name"
        FROM "Models" m
        JOIN "Brands" b ON b."brand_id" = m."brand_id"
        WHERE LOWER(b."brand_name") LIKE LOWER('%ferrari%')
        ORDER BY m."model_name" ASC
        LIMIT 50;

        -- 2) Count customers per brand (through ownership → vins → models → brands)
        SELECT b."brand_name", COUNT(DISTINCT co."customer_id") AS customer_count
        FROM "Customer_Ownership" co
        JOIN "Car_Vins" v ON v."vin" = co."vin"
        JOIN "Models" m ON m."model_id" = v."model_id"
        JOIN "Brands" b ON b."brand_id" = m."brand_id"
        GROUP BY b."brand_name"
        ORDER BY customer_count DESC
        LIMIT 50;

        -- 3) Models using recalled parts (via options → parts)
        SELECT DISTINCT m."model_name"
        FROM "Car_Options" o
        JOIN "Models" m ON m."model_id" = o."model_id"
        JOIN "Car_Parts" p ON p."part_id" IN (o."engine_id", o."transmission_id", o."chassis_id", o."premium_sound_id")
        WHERE p."part_recall" = 1
        ORDER BY m."model_name" ASC;

        FINAL RULE:
        - Return ONLY one ```sql fenced block containing the statement. No extra text.
        """)

        # Keep original formatting prompt as requested.
        self.formatting_prompt = """
            You will be given:
            1) The user's question.
            2) A query result serialized as JSON (list of row objects, possibly empty).

            Goal: State the answer in clear, human-readable prose based ONLY on the given result. Do not mention SQL, databases, schemas, or reasoning.

            Assumptions about result:
            - It is either: [] (no rows), [ {...} ] (single row), or a list of N>=2 rows.
            - Column names are the object keys. Values may be strings, numbers, booleans, or null.

            STRICT OUTPUT RULES
            - If result is empty: output exactly "No relevant data found."
            - If the result is a single scalar (1 row × 1 column): output just that value.
            - If the result is 1 row (<= 8 columns): output one concise sentence in the form "col: value, col: value".
            - If the result is ≤ 10 rows: output a compact bullet list, one item per row.
            - If the result is > 10 rows: summarize — include total row count, the most relevant top items (up to 10), and simple aggregates if present.
            - No markdown tables. Bullets use "- " prefix. No preambles, no epilogues, no extra commentary.

            FORMATTING & PRESENTATION
            - Label selection (multi-row bullets): choose the most human-readable text column as the primary label. Prefer columns named like name, title, brand, model, city, category, or the first non-ID text column. Avoid raw IDs unless no better label exists.
            - Secondary metrics (optional): if numeric columns exist (e.g., count, total, amount, avg, score), append the most relevant one after an em dash: "label — metric_name: value".
            - Sorting:
            * If a clear metric column exists (count, total, amount, revenue, score, avg), sort descending by the strongest metric.
            * Else sort ascending by the chosen label.
            - Nulls: print as "unknown".
            - Booleans: print "Yes"/"No".
            - Numbers: integers as-is; floats to 2 decimals (strip trailing zeros). Use thousands separators for large numbers.
            - Currency: if a column name implies money (price, amount, cost, revenue, total_usd), prefix with "$" and format with thousands separators and 2 decimals.
            - Percentages: if a value is clearly 0–1 and column name implies percent/share/ratio, format as percentage with 1 decimal place.
            - Dates: if values look like YYYY-MM-DD (or ISO), keep "YYYY-MM-DD". Do not use relative dates.
            - Strings longer than 80 chars: truncate with "…".
            - Units in column names (e.g., "_km", "_kg", "_ms"): append unit after value (e.g., "120 ms"). Do not guess units.
            - Do not invent fields or infer data not present. It is allowed to compute simple derived summaries across the given rows: totals, averages, min/max, and percentages-of-total for a single metric column.

            MULTI-ROW SUMMARY (when > 10 rows)
            - First line: "<N> items."
            - If a single dominant numeric metric exists, also print:
            * "total_<metric>: <sum>"
            * "avg_<metric>: <avg>"
            * "min_<metric>: <min>"
            * "max_<metric>: <max>"
            - Then list up to the top 10 items (sorted by that metric or by label if no metric).
            - If a categorical breakdown is obvious (e.g., a column with few distinct values), you may add a one-line tally like "by category — A: x, B: y, C: z".

            EDGE CASES
            - If columns look like a pivot (e.g., many *_count fields), prefer a one-line summary of key totals then bullets for notable highs.
            - If rows look identical on the chosen label, append a differentiating field (e.g., year, city).
            - If only IDs are present, print IDs as the label.

            OUTPUT SHAPES (examples; adapt as per rules)
            - Single scalar:
            "42"
            - Single row:
            "brand_name: Ferrari, model_count: 7"
            - ≤ 10 rows (bullets):
            - "488 GTB — price: $249,900"
            - "Roma — price: $222,620"
            - > 10 rows (summary + top items):
            "37 items. total_amount: $1,238,550, avg_amount: $33,471.62, min_amount: $120.00, max_amount: $98,500.00
            - "Alpha — amount: $98,500.00"
            - "Beta — amount: $76,300.00"
            - …"

            Remember:
            - Be direct and precise.
            - Never mention SQL or internal steps.
            - Use only the provided result.
            """

    def _plan(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self.ollama_client.generate(prompt=user_prompt, system=self.planner_prompt)
            # Extract JSON robustly
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            parsed = json.loads(raw[start:end+1])
            # Basic sanity: ensure tables is a list of known tables (best-effort; rely on generator correctness)
            if not isinstance(parsed, dict) or "tables" not in parsed:
                return None
            return parsed
        except Exception:
            return None

    def _build_system_prompt(self, plan: Optional[Dict[str, Any]]) -> str:
        system = self.system_prompt_base.replace("{dialect}", self.dialect)
        if not plan:
            return system

        def _fmt_list(xs: Any) -> str:
            if not xs:
                return "-"
            if isinstance(xs, list):
                return ", ".join(str(x) for x in xs)
            return str(xs)

        context_hints = textwrap.dedent(f"""
        === CONTEXT HINTS (from structured plan) ===
        action: {_fmt_list(plan.get("action"))}
        tables: {_fmt_list(plan.get("tables"))}
        columns: {_fmt_list(plan.get("columns"))}
        joins: {_fmt_list(plan.get("joins"))}
        filters: {_fmt_list(plan.get("filters"))}
        aggregations: {_fmt_list(plan.get("aggregations"))}
        group_by: {_fmt_list(plan.get("group_by"))}
        order_by: {_fmt_list(plan.get("order_by"))}
        limit: {_fmt_list(plan.get("limit"))}
        date_columns: {_fmt_list(plan.get("date_columns"))}
        constraints_considered: {_fmt_list(plan.get("constraints_considered"))}
        notes: {_fmt_list(plan.get("notes"))}
        === END CONTEXT HINTS ===
        """)

        return system + "\n" + context_hints

    def get_response(self, user_prompt: str, _get: Literal["query", "values", "formatted_response"]= "formatted_response") -> str:
        plan = self._plan(user_prompt)
        print(f"Plan: {plan}")
        system_prompt = self._build_system_prompt(plan)

        last_error: Optional[Exception] = None
        for i in range(3):
            try:
                query = self.ollama_client.generate(prompt=user_prompt, system=system_prompt)
                query = self.sql_utils.extract_query_from_response(query)
                print(f"Generated Query:\n{query}\n")
                if _get == "query":
                    return query
            
                if not self.privileges == "admin":
                    if self.privileges == "read" and ("INSERT" in query.upper() or "UPDATE" in query.upper() or "DELETE" in query.upper()):
                        raise PermissionError("Agent has read-only privileges; cannot execute write operations.")
                    elif self.privileges == "write" and "SELECT" in query.upper():
                        raise PermissionError("Agent has write-only privileges; cannot execute read operations.")

                values = self.sql_utils.execute_query(query)
                print(f"Query Result:\n{values}\n")
                if _get == "values":
                    return values
                fp = f"{user_prompt}\n\nThe Query is: {query}\n\nThe query result is: {values}\n\nProvide the answer now."
                final_response = self.ollama_client.generate(prompt=fp, system=self.formatting_prompt)
                return final_response.strip()
            except Exception as e:
                last_error = e
                print(f"Error executing query: {e} \nRetrying... {i+1}/3\n")
                continue
        return "No relevant data found."
