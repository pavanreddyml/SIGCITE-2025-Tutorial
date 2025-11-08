import os
import sqlite3
import tempfile
import pandas as pd
from IPython.display import display, clear_output
import ipywidgets as w
import matplotlib.pyplot as plt


class DatabaseUI:
    def __init__(self, db_path="/mnt/data/sample.db"):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at: {db_path}")
        self.db_path = db_path

        # State
        self.state = {"current_df": pd.DataFrame(), "current_table": None}

        # Widgets (created in _setup_widgets)
        self.table_dd = None
        self.cols_ms = None
        self.filter_col_dd = None
        self.filter_op_dd = None
        self.filter_val_tb = None
        self.order_by_dd = None
        self.limit_int = None
        self.refresh_btn = None
        self.download_btn = None
        self.reload_schema_btn = None
        self.out = None

        self.x_dd = None
        self.y_dd = None
        self.plot_btn = None
        self.plot_out = None

        self.ui = None

        self._setup_widgets()

    # --- DB helpers ---
    def _get_connection(self):
        # New connection per call ensures fresh view of on-disk DB
        return sqlite3.connect(self.db_path)

    def _list_tables(self):
        with self._get_connection() as con:
            df = pd.read_sql(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;",
                con,
            )
        return df["name"].tolist()

    def _list_columns(self, table):
        with self._get_connection() as con:
            df = pd.read_sql(f"PRAGMA table_info({table});", con)
        return df["name"].tolist(), dict(zip(df["name"], df["type"].fillna("")))

    def _infer_numeric_columns(self, table):
        cols, types = self._list_columns(table)
        numeric = []
        for c in cols:
            t = (types.get(c, "") or "").upper()
            if any(k in t for k in ["INT", "REAL", "FLOA", "DOUB", "NUM"]):
                numeric.append(c)
        return numeric

    # --- Query build/run ---
    @staticmethod
    def _build_query(table, selected_cols, filter_col, filter_op, filter_val, limit, order_by):
        cols_sql = ", ".join([f'"{c}"' for c in selected_cols]) if selected_cols else "*"
        where_sql = ""
        params = []

        if filter_col and filter_op != "(none)" and filter_val not in (None, ""):
            if filter_op in ["contains", "not contains", "starts with", "ends with"]:
                if filter_op == "contains":
                    where_sql = f'WHERE "{filter_col}" LIKE ?'
                    params = [f"%{filter_val}%"]
                elif filter_op == "not contains":
                    where_sql = f'WHERE "{filter_col}" NOT LIKE ?'
                    params = [f"%{filter_val}%"]
                elif filter_op == "starts with":
                    where_sql = f'WHERE "{filter_col}" LIKE ?'
                    params = [f"{filter_val}%"]
                elif filter_op == "ends with":
                    where_sql = f'WHERE "{filter_col}" LIKE ?'
                    params = [f"%{filter_val}"]
            elif filter_op in ["=", "!=", ">", ">=", "<", "<="]:
                where_sql = f'WHERE "{filter_col}" {filter_op} ?'
                params = [filter_val]

        order_sql = f' ORDER BY "{order_by}"' if order_by else ""
        limit_sql = f" LIMIT {int(limit)}" if limit and int(limit) > 0 else ""
        sql = f'SELECT {cols_sql} FROM "{table}" {where_sql}{order_sql}{limit_sql};'
        return sql, params

    def _run_query(self, sql, params=None):
        with self._get_connection() as con:
            return pd.read_sql(sql, con, params=params or [])

    # --- File helper ---
    @staticmethod
    def _save_csv(df):
        path = os.path.join(tempfile.gettempdir(), "db_navigator_view.csv")
        df.to_csv(path, index=False)
        return path

    # --- Widget setup and callbacks ---
    def _setup_widgets(self):
        tables = self._list_tables()
        if not tables:
            raise RuntimeError("No user tables found in the database.")

        self.table_dd = w.Dropdown(options=tables, description="Table:", layout=w.Layout(width="350px"))
        cols_all, _types = self._list_columns(tables[0])
        self.cols_ms = w.SelectMultiple(
            options=cols_all, value=tuple(cols_all), description="Columns", rows=8, layout=w.Layout(width="350px")
        )

        self.filter_col_dd = w.Dropdown(options=[""] + cols_all, description="Filter col:", layout=w.Layout(width="350px"))
        self.filter_op_dd = w.Dropdown(
            options=["(none)", "=", "!=", ">", ">=", "<", "<=", "contains", "not contains", "starts with", "ends with"],
            description="Op:",
            layout=w.Layout(width="200px"),
        )
        self.filter_val_tb = w.Text(description="Value:", layout=w.Layout(width="350px"))
        self.order_by_dd = w.Dropdown(options=[""] + cols_all, description="Order by:", layout=w.Layout(width="350px"))
        self.limit_int = w.BoundedIntText(value=100, min=0, max=1_000_000, step=50, description="LIMIT:", layout=w.Layout(width="200px"))

        self.refresh_btn = w.Button(description="Run", button_style="primary")
        self.download_btn = w.Button(description="Save CSV")
        self.reload_schema_btn = w.Button(description="Reload Schema")
        self.out = w.Output()

        # Plot controls
        self.x_dd = w.Dropdown(options=[""], description="X:", layout=w.Layout(width="250px"))
        self.y_dd = w.Dropdown(options=[""], description="Y:", layout=w.Layout(width="250px"))
        self.plot_btn = w.Button(description="Plot (matplotlib)")
        self.plot_out = w.Output()

        # Bind events
        self.table_dd.observe(self._on_table_change, names="value")
        self.refresh_btn.on_click(self._on_run_clicked)
        self.download_btn.on_click(self._on_download_clicked)
        self.reload_schema_btn.on_click(self._on_reload_schema_clicked)
        self.plot_btn.on_click(self._on_plot_clicked)

        # Layout
        query_box = w.VBox(
            [
                w.HBox([self.table_dd, self.limit_int, self.order_by_dd, self.reload_schema_btn]),
                w.HBox([self.cols_ms]),
                w.HBox([self.filter_col_dd, self.filter_op_dd, self.filter_val_tb]),
                w.HBox([self.refresh_btn, self.download_btn]),
            ]
        )
        plot_box = w.VBox(
            [
                w.HTML("<b>Quick Plot</b>"),
                w.HBox([self.x_dd, self.y_dd, self.plot_btn]),
                self.plot_out,
            ]
        )
        self.ui = w.VBox(
            [
                w.HTML("<h3>SQLite DB Navigator</h3>"),
                query_box,
                self.out,
                w.HTML("<hr>"),
                plot_box,
            ]
        )

        # Initialize
        self.state["current_table"] = self.table_dd.value
        self._on_table_change({"new": self.table_dd.value})

    # --- Callbacks ---
    def _on_table_change(self, change):
        table = change["new"]
        self.state["current_table"] = table
        cols, _ = self._list_columns(table)
        self.cols_ms.options = cols
        self.cols_ms.value = tuple(cols)  # default: all
        self.filter_col_dd.options = [""] + cols
        self.order_by_dd.options = [""] + cols
        with self.out:
            clear_output()
            print(f"Selected table: {table}. Press 'Run' to load data.")

    def _on_reload_schema_clicked(self, _):
        # Re-read tables/columns to pick up external schema changes
        tables = self._list_tables()
        if not tables:
            with self.out:
                clear_output()
                print("No user tables found after reload.")
                return
        current = self.table_dd.value if self.table_dd.value in tables else tables[0]
        self.table_dd.options = tables
        self.table_dd.value = current  # triggers _on_table_change

    def _on_run_clicked(self, _):
        table = self.table_dd.value
        selected_cols = list(self.cols_ms.value)
        filter_col = self.filter_col_dd.value or None
        filter_op = self.filter_op_dd.value
        filter_val = self.filter_val_tb.value
        limit = self.limit_int.value
        order_by = self.order_by_dd.value or None

        sql, params = self._build_query(table, selected_cols, filter_col, filter_op, filter_val, limit, order_by)
        try:
            df = self._run_query(sql, params)
        except Exception as e:
            with self.out:
                clear_output()
                print("SQL:")
                print(sql)
                print(f"\nError: {e}")
            return

        self.state["current_df"] = df

        # Set numeric col candidates for plotting
        numeric_cols = []
        for c in df.columns:
            try:
                pd.to_numeric(df[c], errors="raise")
                numeric_cols.append(c)
            except Exception:
                pass
        if not numeric_cols:
            numeric_cols = self._infer_numeric_columns(table)
            numeric_cols = [c for c in numeric_cols if c in df.columns]

        self.x_dd.options = [""] + numeric_cols
        self.y_dd.options = [""] + numeric_cols

        with self.out:
            clear_output()
            print("SQL:")
            print(sql)
            display(df)

    def _on_download_clicked(self, _):
        df = self.state.get("current_df", pd.DataFrame())
        with self.out:
            if df.empty:
                print("No data to save. Run a query first.")
                return
            path = self._save_csv(df)
            clear_output(wait=False)
            print(f"Saved current view to: {path}")

    def _on_plot_clicked(self, _):
        df = self.state.get("current_df", pd.DataFrame())
        with self.plot_out:
            clear_output()
            if df.empty:
                print("No data to plot. Run a query first.")
                return
            x = self.x_dd.value
            y = self.y_dd.value
            if not x or not y:
                print("Pick X and Y from numeric columns.")
                return
            xd = pd.to_numeric(df[x], errors="coerce")
            yd = pd.to_numeric(df[y], errors="coerce")
            m = xd.notna() & yd.notna()
            if m.sum() == 0:
                print("No numeric data to plot after conversion.")
                return
            plt.figure()
            plt.scatter(xd[m], yd[m])
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f"{self.table_dd.value}: {y} vs {x}")
            plt.show()

    # --- Public API ---
    def run(self):
        display(self.ui)
        # Show initial status for the currently selected table
        self._on_table_change({"new": self.table_dd.value})