# Colab-friendly HTML/Widget app for your specific auto-sales schema.
# Usage:
#   app = CarDealerApp(db_path="/mnt/data/sample.db")
#   app.run()

import os
import sqlite3
import pandas as pd
from datetime import datetime
from IPython.display import display, clear_output, HTML
import ipywidgets as w

class CarDealerApp:
    """
    A minimal, schema-aware UI for exploring and demoing a car dealership database.
    Designed for Google Colab / Jupyter. Focuses on "shopping" and "purchase" flows.

    Expected schema tables (per your description):
      - Brands(brand_id PK, brand_name)
      - Models(model_id PK, brand_id FK, ... name column)
      - Car_Options(option_set_id PK, model_id FK, engine_id FK, transmission_id FK, chassis_id FK,
                    premium_sound_id FK, color, option_set_price)
      - Car_Vins(vin PK, model_id FK, option_set_id FK, manufactured_date, manufactured_plant_id FK)
      - Customers(customer_id PK, first_name, last_name, gender, household_income, birthdate,
                  phone_number, email) with gender CHECK('Male' or 'Female')
      - Dealers(dealer_id PK, dealer_name, dealer_address)
      - Dealer_Brand(dealer_id, brand_id) PK(dealer_id, brand_id)
      - Manufacture_Plant(…)
      - Car_Parts(…)  # only referenced; not strictly required to view/purchase

    Notes
    - The app enforces SQLite foreign_keys; integrity errors will surface clearly in the UI.
    - "Available" cars are VINs not yet present in Customer_Ownership.
    - Purchase flow can create a new customer or reference an existing one.
    - This class only builds the UI. It will use whatever DB contents you provide.
    """

    # ------------------ lifecycle ------------------
    def __init__(self, db_path="/mnt/data/sample.db"):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at: {db_path}")
        self.db_path = db_path

        # State cache
        self._state = dict(
            selected_brand_id=None,
            selected_model_id=None,
            selected_option_set_id=None,
            selected_vin=None,
            selected_dealer_id=None,
            existing_customer_id=None,
            purchase_price=None,
            purchase_date=datetime.today().strftime("%Y-%m-%d"),
        )

        # Widgets
        self._build_widgets()
        self._wire_events()
        self._layout()

        # Initial load
        self._refresh_brands()
        self._refresh_dealers()
        self._refresh_customers()
        self._update_shop_tables()

    def run(self):
        display(self.root)

    # ------------------ DB helpers ------------------
    def _connect(self):
        con = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        con.execute("PRAGMA foreign_keys=ON;")
        return con

    def _table_exists(self, name):
        with self._connect() as con:
            cur = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;",
                (name,)
            )
            return cur.fetchone() is not None

    def _df(self, sql, params=()):
        with self._connect() as con:
            return pd.read_sql(sql, con, params=params)

    def _exec(self, sql, params=()):
        with self._connect() as con:
            cur = con.execute(sql, params)
            con.commit()
            return cur.lastrowid

    def _has_column(self, table, col):
        with self._connect() as con:
            df = pd.read_sql(f"PRAGMA table_info({table});", con)
        return (df["name"] == col).any()

    def _best_name_col(self, table, fallbacks=("model_name","name","label","title")):
        with self._connect() as con:
            cols = pd.read_sql(f"PRAGMA table_info({table});", con)["name"].tolist()
        for fb in fallbacks:
            if fb in cols:
                return fb
        # fallback to first text-like column if any
        return next((c for c in cols if "name" in c.lower()), cols[0] if cols else "rowid")

    # ------------------ Data queries ------------------
    def _get_brands(self):
        if not self._table_exists("Brands"):
            return pd.DataFrame(columns=["brand_id","brand_name"])
        return self._df("SELECT brand_id, brand_name FROM Brands ORDER BY brand_name;")

    def _get_models_by_brand(self, brand_id):
        if not self._table_exists("Models"):
            return pd.DataFrame(columns=["model_id","brand_id"])
        name_col = self._best_name_col("Models")
        df = self._df(
            f"SELECT model_id, brand_id, \"{name_col}\" AS model_name FROM Models WHERE brand_id=? ORDER BY model_name;",
            (brand_id,)
        )
        return df

    def _get_options_by_model(self, model_id):
        if not self._table_exists("Car_Options"):
            return pd.DataFrame(columns=["option_set_id","model_id","color","option_set_price"])
        sql = """
            SELECT option_set_id, model_id, color, option_set_price,
                   engine_id, transmission_id, chassis_id, premium_sound_id
            FROM Car_Options
            WHERE model_id=?
            ORDER BY option_set_price, color;
        """
        return self._df(sql, (model_id,))

    def _get_available_vins(self, model_id=None, option_set_id=None):
        if not self._table_exists("Car_Vins"):
            return pd.DataFrame(columns=["vin","model_id","option_set_id","manufactured_date","manufactured_plant_id"])
        base = """
            SELECT v.vin, v.model_id, v.option_set_id, v.manufactured_date, v.manufactured_plant_id
            FROM Car_Vins v
            LEFT JOIN Customer_Ownership co ON co.vin = v.vin
            WHERE co.vin IS NULL
        """
        params = []
        if model_id:
            base += " AND v.model_id=?"
            params.append(model_id)
        if option_set_id:
            base += " AND v.option_set_id=?"
            params.append(option_set_id)
        base += " ORDER BY v.vin;"
        return self._df(base, tuple(params))

    def _get_dealers(self):
        if not self._table_exists("Dealers"):
            return pd.DataFrame(columns=["dealer_id","dealer_name","dealer_address"])
        return self._df("SELECT dealer_id, dealer_name, dealer_address FROM Dealers ORDER BY dealer_name;")

    def _get_dealers_for_brand(self, brand_id):
        if not (self._table_exists("Dealer_Brand") and self._table_exists("Dealers")):
            return self._get_dealers()
        sql = """
            SELECT d.dealer_id, d.dealer_name, d.dealer_address
            FROM Dealers d
            JOIN Dealer_Brand db ON db.dealer_id = d.dealer_id
            WHERE db.brand_id=?
            ORDER BY d.dealer_name;
        """
        return self._df(sql, (brand_id,))

    def _get_customers(self):
        if not self._table_exists("Customers"):
            return pd.DataFrame(columns=["customer_id","first_name","last_name"])
        cols = ["customer_id","first_name","last_name","gender","household_income","birthdate","phone_number","email"]
        have = [c for c in cols if self._has_column("Customers", c)]
        sql = "SELECT " + ", ".join(f"\"{c}\"" for c in have) + " FROM Customers ORDER BY last_name, first_name;"
        return self._df(sql)

    def _get_inventory_snapshot(self):
        # Useful for the Inventory tab: join Brands->Models->Car_Options->Car_Vins (available)
        if not (self._table_exists("Brands") and self._table_exists("Models") and self._table_exists("Car_Options") and self._table_exists("Car_Vins")):
            return pd.DataFrame(columns=["vin","brand","model","color","price","manufactured_date"])
        model_name = self._best_name_col("Models")
        sql = f"""
        WITH available AS (
          SELECT v.*
          FROM Car_Vins v
          LEFT JOIN Customer_Ownership co ON co.vin = v.vin
          WHERE co.vin IS NULL
        )
        SELECT a.vin,
               b.brand_name AS brand,
               m."{model_name}" AS model,
               o.color,
               o.option_set_price AS price,
               a.manufactured_date
        FROM available a
        JOIN Models m ON m.model_id = a.model_id
        JOIN Brands b ON b.brand_id = m.brand_id
        JOIN Car_Options o ON o.option_set_id = a.option_set_id
        ORDER BY b.brand_name, model, o.option_set_price, a.vin;
        """
        return self._df(sql)

    # ------------------ Purchase ops ------------------
    def _create_customer(self, payload):
        # payload keys: first_name, last_name, gender ('Male'/'Female' or ''), household_income (int/None),
        #               birthdate (YYYY-MM-DD), phone_number (int), email
        cols = []
        vals = []
        params = []
        for k in ["first_name","last_name","gender","household_income","birthdate","phone_number","email"]:
            if self._has_column("Customers", k):
                cols.append(k)
                vals.append("?")
                params.append(payload.get(k))
        sql = f"INSERT INTO Customers ({', '.join(cols)}) VALUES ({', '.join(vals)});"
        return self._exec(sql, tuple(params))

    def _insert_ownership(self, customer_id, vin, dealer_id, price, purchase_date, warantee_expire_date=None):
        cols = ["customer_id","vin","purchase_date","purchase_price","dealer_id"]
        vals = ["?","?","?","?","?"]
        params = [customer_id, vin, purchase_date, price, dealer_id]
        if self._has_column("Customer_Ownership", "warantee_expire_date"):
            cols.append("warantee_expire_date")
            vals.append("?")
            params.append(warantee_expire_date)
        sql = f"INSERT INTO Customer_Ownership ({', '.join(cols)}) VALUES ({', '.join(vals)});"
        self._exec(sql, tuple(params))

    # ------------------ UI building ------------------
    def _build_widgets(self):
        # Headline
        self.header = w.HTML("<h2>🏁 Car Dealer Demo (SQLite)</h2>")

        # Shop panel controls
        self.brand_dd = w.Dropdown(options=[], description="Brand:", layout=w.Layout(width="280px"))
        self.model_dd = w.Dropdown(options=[], description="Model:", layout=w.Layout(width="280px"))
        self.option_dd = w.Dropdown(options=[], description="Option set:", layout=w.Layout(width="320px"))
        self.vin_dd = w.Dropdown(options=[], description="VIN:", layout=w.Layout(width="280px"))
        self.dealer_dd = w.Dropdown(options=[], description="Dealer:", layout=w.Layout(width="360px"))

        self.price_tb = w.IntText(description="Price:", layout=w.Layout(width="220px"))
        self.date_tb = w.Text(value=datetime.today().strftime("%Y-%m-%d"), description="Date:", layout=w.Layout(width="220px"))
        self.warranty_tb = w.Text(placeholder="YYYY-MM-DD (optional)", description="Warranty:", layout=w.Layout(width="280px"))

        self.purchase_btn = w.Button(description="Complete Purchase", button_style="success")
        self.refresh_btn = w.Button(description="Refresh Lists", button_style="primary")
        self.out_shop = w.Output()

        # Existing vs New customer
        self.customer_mode = w.ToggleButtons(options=[("Use existing","existing"), ("Create new","new")], value="existing")
        self.existing_customer_dd = w.Dropdown(options=[], description="Customer:", layout=w.Layout(width="420px"))

        self.new_first = w.Text(description="First:", layout=w.Layout(width="250px"))
        self.new_last = w.Text(description="Last:", layout=w.Layout(width="250px"))
        self.new_gender = w.Dropdown(options=["","Male","Female"], description="Gender:", layout=w.Layout(width="200px"))
        self.new_income = w.IntText(description="Income:", layout=w.Layout(width="220px"))
        self.new_birth = w.Text(placeholder="YYYY-MM-DD", description="Birthdate:", layout=w.Layout(width="260px"))
        self.new_phone = w.Text(description="Phone:", layout=w.Layout(width="240px"))
        self.new_email = w.Text(description="Email:", layout=w.Layout(width="320px"))

        # Shop tables
        self.tbl_brands = w.Output(layout=w.Layout(border='1px solid #ddd'))
        self.tbl_models = w.Output(layout=w.Layout(border='1px solid #ddd'))
        self.tbl_options = w.Output(layout=w.Layout(border='1px solid #ddd'))
        self.tbl_vins = w.Output(layout=w.Layout(border='1px solid #ddd'))

        # Inventory tab
        self.inv_refresh = w.Button(description="Refresh Inventory")
        self.inv_out = w.Output()

        # Customers tab
        self.cust_refresh = w.Button(description="Refresh Customers")
        self.cust_out = w.Output()

        # Dealers tab
        self.dealer_refresh = w.Button(description="Refresh Dealers")
        self.dealer_out = w.Output()

        # SQL tab (read-only helper)
        self.sql_txt = w.Textarea(placeholder="SELECT ...", layout=w.Layout(width="100%", height="140px"))
        self.sql_run = w.Button(description="Run SELECT")
        self.sql_out = w.Output()

        # Footer
        self.footer = w.HTML(
            "<small>Tip: Intentionally break FKs or CHECKs in another cell and then try actions here to surface errors.</small>"
        )

    def _wire_events(self):
        self.refresh_btn.on_click(self._on_refresh_all)
        self.purchase_btn.on_click(self._on_purchase)

        self.brand_dd.observe(self._on_brand_change, names="value")
        self.model_dd.observe(self._on_model_change, names="value")
        self.option_dd.observe(self._on_option_change, names="value")

        self.customer_mode.observe(self._on_customer_mode_change, names="value")

        self.inv_refresh.on_click(lambda _: self._render_inventory())
        self.cust_refresh.on_click(lambda _: self._render_customers())
        self.dealer_refresh.on_click(lambda _: self._render_dealers())

        self.sql_run.on_click(self._on_sql_run)

    def _layout(self):
        # Shop section (left controls)
        left = w.VBox([
            w.HTML("<b>Choose a car</b>"),
            self.brand_dd,
            self.model_dd,
            self.option_dd,
            self.vin_dd,
            self.dealer_dd,
            w.HTML("<hr><b>Buyer</b>"),
            self.customer_mode,
            w.HBox([self.existing_customer_dd]),
            w.HBox([self.new_first, self.new_last]),
            w.HBox([self.new_gender, self.new_income]),
            w.HBox([self.new_birth, self.new_phone]),
            w.HBox([self.new_email]),
            w.HTML("<hr><b>Purchase</b>"),
            w.HBox([self.price_tb, self.date_tb, self.warranty_tb]),
            w.HBox([self.purchase_btn, self.refresh_btn]),
            self.out_shop
        ], layout=w.Layout(width="52%"))

        # Shop section (right data previews)
        right = w.VBox([
            w.HTML("<b>Brands</b>"), self.tbl_brands,
            w.HTML("<b>Models (selected brand)</b>"), self.tbl_models,
            w.HTML("<b>Option Sets (selected model)</b>"), self.tbl_options,
            w.HTML("<b>Available VINs (filtered)</b>"), self.tbl_vins,
        ], layout=w.Layout(width="48%"))

        shop = w.HBox([left, right])

        # Inventory tab
        inventory = w.VBox([self.inv_refresh, self.inv_out])

        # Customers tab
        customers = w.VBox([self.cust_refresh, self.cust_out])

        # Dealers tab
        dealers = w.VBox([self.dealer_refresh, self.dealer_out])

        # SQL tab
        sql = w.VBox([self.sql_txt, self.sql_run, self.sql_out])

        self.tabs = w.Tab(children=[shop, inventory, customers, dealers, sql])
        for i, title in enumerate(["Shop", "Inventory", "Customers", "Dealers", "SQL (SELECT only)"]):
            self.tabs.set_title(i, title)

        self.root = w.VBox([self.header, self.tabs, self.footer])

        # default states
        self._set_customer_fields_visibility()

    # ------------------ Events ------------------
    def _on_refresh_all(self, _):
        self._refresh_brands()
        self._refresh_dealers()
        self._refresh_customers()
        self._update_shop_tables()
        with self.out_shop:
            clear_output()
            print("Lists refreshed.")

    def _on_brand_change(self, change):
        brand_id = change["new"]
        self._state["selected_brand_id"] = brand_id
        self._refresh_models(brand_id)
        self._refresh_dealers_for_brand(brand_id)
        self._refresh_options(None)
        self._refresh_vins()
        self._update_shop_tables()

    def _on_model_change(self, change):
        model_id = change["new"]
        self._state["selected_model_id"] = model_id
        self._refresh_options(model_id)
        self._refresh_vins()
        self._update_shop_tables()

    def _on_option_change(self, change):
        option_set_id = change["new"]
        self._state["selected_option_set_id"] = option_set_id
        self._refresh_vins(option_set_id=option_set_id)
        self._update_shop_tables()

    def _on_customer_mode_change(self, change):
        self._set_customer_fields_visibility()

    def _on_purchase(self, _):
        with self.out_shop:
            clear_output()
            try:
                # Validate selections
                vin = self.vin_dd.value
                dealer_id = self.dealer_dd.value
                price = self.price_tb.value
                date_str = self.date_tb.value.strip()
                warranty = self.warranty_tb.value.strip() or None
                if not vin:
                    print("Select a VIN.")
                    return
                if not dealer_id:
                    print("Select a dealer.")
                    return
                if price is None or price <= 0:
                    print("Enter a valid purchase price (>0).")
                    return
                # basic date validation (sqlite will also validate if constraints exist)
                for d in [date_str] + ([warranty] if warranty else []):
                    if d and not self._is_iso_date(d):
                        print(f"Invalid date: {d} (expected YYYY-MM-DD).")
                        return

                # Resolve / create customer
                if self.customer_mode.value == "existing":
                    customer_id = self.existing_customer_dd.value
                    if not customer_id:
                        print("Pick an existing customer.")
                        return
                else:
                    payload = dict(
                        first_name=self.new_first.value.strip(),
                        last_name=self.new_last.value.strip(),
                        gender=self.new_gender.value or None,
                        household_income=(self.new_income.value if self.new_income.value else None),
                        birthdate=self.new_birth.value.strip() or None,
                        phone_number=self._safe_int(self.new_phone.value),
                        email=self.new_email.value.strip() or None,
                    )
                    if not payload["first_name"] or not payload["last_name"]:
                        print("First and Last name are required for new customer.")
                        return
                    # gender check to help surface your table CHECK sooner
                    if payload["gender"] not in (None, "", "Male", "Female"):
                        print("Gender must be 'Male' or 'Female' (or leave blank if column allows NULL).")
                        return
                    customer_id = self._create_customer(payload)
                    print(f"Created customer_id={customer_id}")

                # Insert ownership
                self._insert_ownership(
                    customer_id=customer_id,
                    vin=vin,
                    dealer_id=dealer_id,
                    price=price,
                    purchase_date=date_str,
                    warantee_expire_date=warranty
                )
                print(f"✅ Purchase recorded: customer {customer_id} -> VIN {vin}.")

                # Refresh UI lists (VIN becomes unavailable)
                self._refresh_customers()
                self._refresh_vins()
                self._render_inventory()

            except Exception as e:
                # Integrity violations (FK/CHECK/PK) will bubble up here
                print("❌ Error during purchase:")
                print(str(e))

    def _on_sql_run(self, _):
        q = (self.sql_txt.value or "").strip()
        with self.sql_out:
            clear_output()
            if not q:
                print("Enter a SELECT query.")
                return
            if not q.lower().lstrip().startswith("select"):
                print("Read-only: only SELECT is allowed here.")
                return
            try:
                df = self._df(q)
                if df.empty:
                    print("(no rows)")
                else:
                    display(df)
            except Exception as e:
                print("Error:", str(e))

    # ------------------ UI refresh/render helpers ------------------
    def _refresh_brands(self):
        dfb = self._get_brands()
        options = [(f"{r.brand_name} (#{r.brand_id})", int(r.brand_id)) for _, r in dfb.iterrows()]
        self.brand_dd.options = options or [("— none —", None)]
        if options:
            self.brand_dd.value = options[0][1]
        self._render_table(self.tbl_brands, dfb)

    def _refresh_models(self, brand_id):
        dfm = self._get_models_by_brand(brand_id) if brand_id else pd.DataFrame(columns=["model_id","brand_id","model_name"])
        options = [(f"{r.model_name} (#{r.model_id})", int(r.model_id)) for _, r in dfm.iterrows()]
        self.model_dd.options = options or [("— none —", None)]
        self.model_dd.value = options[0][1] if options else None
        self._render_table(self.tbl_models, dfm)

    def _refresh_options(self, model_id):
        dfo = self._get_options_by_model(model_id) if model_id else pd.DataFrame(columns=["option_set_id"])
        def opt_label(r):
            parts = [f"id={r.option_set_id}"]
            if pd.notna(r.get("color", None)):
                parts.append(f"color={r.color}")
            if pd.notna(r.get("option_set_price", None)):
                parts.append(f"price={r.option_set_price}")
            return ", ".join(parts)
        options = [(opt_label(r), int(r.option_set_id)) for _, r in dfo.iterrows()]
        self.option_dd.options = options or [("— any option —", None)]
        self.option_dd.value = options[0][1] if options else None
        self._render_table(self.tbl_options, dfo)

    def _refresh_vins(self, option_set_id=None):
        mid = self.model_dd.value
        dfv = self._get_available_vins(model_id=mid, option_set_id=option_set_id or None)
        options = [(f"{int(r.vin)}", int(r.vin)) for _, r in dfv.iterrows()] if "vin" in dfv.columns else []
        self.vin_dd.options = options or [("— none available —", None)]
        self.vin_dd.value = options[0][1] if options else None
        self._render_table(self.tbl_vins, dfv)

    def _refresh_dealers(self):
        dfd = self._get_dealers()
        options = [(f"{r.dealer_name} (#{r.dealer_id})", int(r.dealer_id)) for _, r in dfd.iterrows()]
        self.dealer_dd.options = options or [("— none —", None)]
        self.dealer_dd.value = options[0][1] if options else None
        self._render_dealers()

    def _refresh_dealers_for_brand(self, brand_id):
        dfd = self._get_dealers_for_brand(brand_id) if brand_id else self._get_dealers()
        options = [(f"{r.dealer_name} (#{r.dealer_id})", int(r.dealer_id)) for _, r in dfd.iterrows()]
        self.dealer_dd.options = options or [("— none —", None)]
        self.dealer_dd.value = options[0][1] if options else None
        self._render_dealers(dfd)

    def _refresh_customers(self):
        dfc = self._get_customers()
        def label(r):
            base = f"{r.get('last_name','')}, {r.get('first_name','')} (#{r.customer_id})"
            if pd.notna(r.get("email", None)) and str(r.get("email")).strip():
                base += f" · {r.get('email')}"
            return base
        options = [(label(r), int(r.customer_id)) for _, r in dfc.iterrows()] if not dfc.empty else [("— none —", None)]
        self.existing_customer_dd.options = options
        self.existing_customer_dd.value = options[0][1] if options and options[0][1] is not None else None
        self._render_customers(dfc)

    def _update_shop_tables(self):
        # full refresh of right side tables for current selections
        # Already rendered within individual refresh calls; keep this method for clarity/extensibility.
        pass

    def _render_inventory(self):
        dfi = self._get_inventory_snapshot()
        with self.inv_out:
            clear_output()
            if dfi.empty:
                print("(no available inventory)")
            else:
                display(dfi)

    def _render_customers(self, df=None):
        df = df if df is not None else self._get_customers()
        with self.cust_out:
            clear_output()
            if df.empty:
                print("(no customers)")
            else:
                display(df)

    def _render_dealers(self, df=None):
        df = df if df is not None else self._get_dealers()
        with self.dealer_out:
            clear_output()
            if df.empty:
                print("(no dealers)")
            else:
                display(df)

    def _render_table(self, out_widget, df):
        with out_widget:
            clear_output()
            if df is None or df.empty:
                print("(empty)")
            else:
                display(df)

    def _set_customer_fields_visibility(self):
        is_new = (self.customer_mode.value == "new")
        self.existing_customer_dd.layout.display = ("" if not is_new else "none")
        for wid in [self.new_first, self.new_last, self.new_gender, self.new_income, self.new_birth, self.new_phone, self.new_email]:
            wid.layout.display = ("" if is_new else "none")

    # ------------------ utils ------------------
    @staticmethod
    def _is_iso_date(s):
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return True
        except Exception:
            return False

    @staticmethod
    def _safe_int(s):
        try:
            return int(str(s).strip())
        except Exception:
            return None
