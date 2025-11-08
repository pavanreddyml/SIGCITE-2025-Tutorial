from __future__ import annotations

import importlib
import inspect
import json
import ast
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple
from typing import Literal, get_args, get_origin

import ipywidgets as widgets
from ipywidgets import Layout

from PIL import Image
import numpy as np
from io import BytesIO

CATEGORY_PACKAGES: Mapping[str, str] = {
    "whitebox": "adversarial_lab.arsenal.adversarial.whitebox",
    "blackbox": "adversarial_lab.arsenal.adversarial.blackbox",
}

WHITEBOX_MODELS: Tuple[str, ...] = ("inception", "resnet", "mobilenet")
BLACKBOX_MODELS: Tuple[str, ...] = ("mnist_digits",)

MNIST_DIGITS_CLASSES = tuple(str(i) for i in range(10))


@dataclass(frozen=True)
class AttackInfo:
    """Container describing an attack implementation."""
    label: str
    cls: type


class AttackSelectorUI:
    def __init__(self, root_path: Optional[Path | str] = None) -> None:
        self.root_path = Path(root_path or Path.cwd())
        self._attacks: Dict[str, Dict[str, AttackInfo]] = {}
        self._attack_param_widgets: List[widgets.Widget] = []
        # cache of [(name, idx)]
        self._imagenet_pairs_cache: Optional[List[Tuple[str, int]]] = None
        self._uploaded_image_array: Optional[np.ndarray] = None

        common_style = {"description_width": "initial"}
        # shift right so labels aren’t cut off
        indent_layout = Layout(margin="0 0 0 220px")

        # ---- Top: Category ----
        self.category_selector = widgets.ToggleButtons(
            options=[("Whitebox", "whitebox"), ("Blackbox", "blackbox")],
            description="Category:",
            value="whitebox",
            button_style="",
            style=common_style,
            layout=indent_layout,
        )

        # ---- Attack + params ----
        self.attack_selector = widgets.Dropdown(
            description="Attack:",
            style=common_style,
            layout=indent_layout,
        )
        self.param_container = widgets.VBox(layout=indent_layout)

        # ---- Dataset section ----
        self.model_header = widgets.HTML(
            value="<h4 style='margin:8px 0 0 220px;'>Model</h4>"
        )

        self.model_selector = widgets.Dropdown(
            description="Model:",
            style=common_style,
            layout=indent_layout,
        )

        # ---- Image source selector (two-box style like Category) ----
        self.image_source_selector = widgets.ToggleButtons(
            options=[("Dataset Image", "dataset"), ("Upload Image", "upload")],
            description="Image Source:",
            value="dataset",
            button_style="",
            style=common_style,
            layout=indent_layout,
        )

        # ---- Image selection (dataset-based) ----
        self.image_class_selector = widgets.Dropdown(
            description="Image Class:",
            style=common_style,
            layout=indent_layout,
        )
        self.image_name_selector = widgets.Dropdown(
            description="Image Name:",
            style=common_style,
            layout=indent_layout,
        )

        # ---- Custom image upload (optional) ----
        self.custom_image_uploader = widgets.FileUpload(
            accept="image/*",
            multiple=False,
            description="Upload Image",
            style=common_style,
            layout=indent_layout,
        )
        self._custom_image_help = widgets.HTML(
            value="<em style='margin-left:220px;'>If provided, this image overrides Image Class/Name.</em>"
        )

        # Container that swaps between dataset selectors and upload widget
        self.image_source_container = widgets.VBox(layout=Layout(margin="0"))

        # ---- Target class (show 'idx - label') ----
        self.target_class_selector = widgets.Dropdown(
            description="Target Class Idx:",
            style=common_style,
            layout=indent_layout,
        )

        self._attach_observers()
        self._initialise_state()

        # Assemble in requested order (image source selector/area below "Model"):
        self.widget = widgets.VBox(
            [
                self.category_selector,          # Category
                self.attack_selector,            # Attack selector
                # Attack params (rendered dynamically)
                self.param_container,
                widgets.HTML("<hr />"),
                self.model_header,               # Dataset section header
                self.model_selector,             # Model
                self.image_source_selector,      # Image Source (two-box style)
                self.image_source_container,     # Either dataset selectors or upload widget
                # Target class idx - label (at the end)
                self.target_class_selector,
            ]
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def display(self) -> None:
        from IPython.display import display
        display(self.widget)

    def get_configuration(self) -> Dict[str, Any]:
        # Numeric idx as value; label is display only
        target_val = self.target_class_selector.value
        use_upload = self.image_source_selector.value == "upload"
        image_array = self._uploaded_image_array if use_upload else None

        return {
            "category": self.category_selector.value,
            "model": self.model_selector.value,           # Selector expects "model"
            "attack": self.attack_selector.value,
            "attack_params": self._collect_attack_parameters(),
            "image_class": None if use_upload else self.image_class_selector.value,
            "image_name": None if use_upload else self.image_name_selector.value,
            "image_array": image_array,                   # Preferred by Selector if not None
            "target_class_idx": target_val,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _attach_observers(self) -> None:
        self.category_selector.observe(self._on_category_change, names="value")
        self.model_selector.observe(self._on_model_change, names="value")
        self.attack_selector.observe(self._on_attack_change, names="value")
        self.image_class_selector.observe(
            self._on_image_class_change, names="value")
        self.image_source_selector.observe(
            self._on_image_source_change, names="value")
        self.custom_image_uploader.observe(
            self._on_custom_image_upload, names="value")

    def _initialise_state(self) -> None:
        self._refresh_models()
        self._refresh_attacks()
        self._refresh_datasets()
        self._sync_image_source_area()

    def _on_category_change(self, change: MutableMapping[str, Any]) -> None:
        if change.get("new") == change.get("old"):
            return
        self._refresh_models()
        self._refresh_attacks()
        self._imagenet_pairs_cache = None  # reset on category switch
        self._refresh_datasets()

    def _on_model_change(self, change: MutableMapping[str, Any]) -> None:
        if change.get("new") == change.get("old"):
            return
        self._refresh_datasets()

    def _on_attack_change(self, change: MutableMapping[str, Any]) -> None:
        if change.get("new") == change.get("old"):
            return
        self._render_attack_parameters()

    def _on_image_class_change(self, change: MutableMapping[str, Any]) -> None:
        if change.get("new") == change.get("old"):
            return
        self._refresh_image_names()

    def _on_image_source_change(self, change: MutableMapping[str, Any]) -> None:
        if change.get("new") == change.get("old"):
            return
        if change.get("new") == "upload":
            self.image_class_selector.value = None
            self.image_name_selector.value = None
        self._sync_image_source_area()

    def _on_custom_image_upload(self, change: MutableMapping[str, Any]) -> None:
        fileinfo = self._get_first_upload(change.get("new"))
        if not fileinfo:
            self._uploaded_image_array = None
            return
        img = Image.open(BytesIO(fileinfo["content"])).convert("RGB")
        self._uploaded_image_array = np.array(img)

    def _get_first_upload(self, value) -> Optional[dict]:
        if not value:
            return None
        if isinstance(value, dict):
            return next(iter(value.values())) if value else None
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return None

    def _sync_image_source_area(self) -> None:
        use_upload = self.image_source_selector.value == "upload"
        if use_upload:
            self.image_class_selector.disabled = True
            self.image_name_selector.disabled = True
            self.image_source_container.children = [
                self.custom_image_uploader, self._custom_image_help]
        else:
            self.image_class_selector.disabled = False
            self.image_name_selector.disabled = False
            self.image_source_container.children = [
                self.image_class_selector, self.image_name_selector]

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------
    def _refresh_models(self) -> None:
        category = self.category_selector.value
        if category == "whitebox":
            options = [(model.title(), model) for model in WHITEBOX_MODELS]
        else:
            options = [(model.replace("_", " ").title(), model)
                       for model in BLACKBOX_MODELS]

        self.model_selector.options = options
        self.model_selector.value = options[0][1] if options else None

    def _dataset_root(self) -> Optional[Path]:
        category = self.category_selector.value
        if category == "whitebox":
            candidate = self._first_existing(
                self.root_path / "assets" / "images" / "imagenet",
                self.root_path / "examples" / "data" / "imagenet",
            )
        else:
            model = self.model_selector.value or ""
            if model == "mnist_digits":
                candidate = self._first_existing(
                    self.root_path / "assets" / "images" / "digits",
                    self.root_path / "examples" / "data" / "digits",
                )
            else:
                candidate = None
        if candidate and candidate.exists():
            return candidate
        return None

    def _refresh_datasets(self) -> None:
        dataset_root = self._dataset_root()
        classes = self._list_directory_names(dataset_root)

        # Image class dropdown (directory names if present)
        class_options = self._build_options(
            classes, empty_label="No classes available")
        self.image_class_selector.options = class_options
        if len(class_options) > 1:
            self.image_class_selector.value = class_options[1][1]
        elif class_options:
            self.image_class_selector.value = class_options[0][1]
        else:
            self.image_class_selector.value = None

        # Target class dropdown (IDX + label)
        target_idx_options = self._target_class_options(
            category=self.category_selector.value)
        self.target_class_selector.options = target_idx_options
        self.target_class_selector.value = (
            target_idx_options[1][1] if len(target_idx_options) > 1 else (
                target_idx_options[0][1] if target_idx_options else None)
        )

        self._refresh_image_names()

    def _refresh_image_names(self) -> None:
        dataset_root = self._dataset_root()
        class_name = self.image_class_selector.value
        image_dir = dataset_root / class_name if dataset_root and class_name else None
        image_names = self._list_file_names(image_dir)

        image_options = self._build_options(
            image_names, empty_label="No images available")
        self.image_name_selector.options = image_options
        if len(image_options) > 1:
            self.image_name_selector.value = image_options[1][1]
        elif image_options:
            self.image_name_selector.value = image_options[0][1]
        else:
            self.image_name_selector.value = None

    def _target_class_options(self, category: str) -> List[Tuple[str, Optional[int]]]:
        """Return dropdown options as [('idx - label', idx), ...]."""
        options: List[Tuple[str, int]] = []
        if category == "whitebox":
            # [(name, idx), ...]
            pairs = self._imagenet_idx_label_pairs_from_tf()
            options = [(f"{idx} - {name}", idx) for (name, idx) in pairs]
        else:
            model = self.model_selector.value or ""
            if model == "mnist_digits":
                options = [(f"{i} - {name}", i)
                           for i, name in enumerate(MNIST_DIGITS_CLASSES)]
            else:
                options = []

        if not options:
            return [("No target classes available", None)]
        return [("Select...", None)] + options

    def _imagenet_idx_label_pairs_from_tf(self) -> List[Tuple[str, int]]:
        """Load ImageNet (ILSVRC-2012) class names and indices via TensorFlow Keras.

        Returns a list of (name, idx) sorted by idx.
        """
        if self._imagenet_pairs_cache is not None:
            return self._imagenet_pairs_cache

        from tensorflow.keras.utils import get_file
        json_path = get_file(
            fname="imagenet_class_index.json",
            origin="https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
            cache_subdir="models",
            file_hash=None,
        )
        with open(json_path, "r") as f:
            class_index: Dict[str, List[str]] = json.load(
                f)  # {"0": ["n01440764","tench"], ...}

        items: List[Tuple[str, int]] = []
        for k, (_wnid, name) in class_index.items():
            idx = int(k)
            items.append((name, idx))
        items.sort(key=lambda x: x[1])

        self._imagenet_pairs_cache = items
        return items

    # ------------------------------------------------------------------
    # Attack helpers
    # ------------------------------------------------------------------
    def _refresh_attacks(self) -> None:
        category = self.category_selector.value
        attacks = self._discover_attacks(category)
        options = self._build_options(
            attacks.keys(), empty_label="No attacks found")
        self.attack_selector.options = options
        if len(options) > 1:
            self.attack_selector.value = options[1][1]
        elif options:
            self.attack_selector.value = options[0][1]
        else:
            self.attack_selector.value = None
        self._render_attack_parameters()

    def _render_attack_parameters(self) -> None:
        selected = self.attack_selector.value
        category = self.category_selector.value
        attack_info = self._attacks.get(category, {}).get(selected)

        if not attack_info:
            self.param_container.children = [
                widgets.HTML("<em>No attack selected.</em>")]
            self._attack_param_widgets = []
            return

        widgets_list = self._create_widgets_for_attack(attack_info.cls)
        self._attack_param_widgets = widgets_list
        if widgets_list:
            self.param_container.children = widgets_list
        else:
            self.param_container.children = [widgets.HTML(
                "<em>No configurable parameters.</em>")]

    def _collect_attack_parameters(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for widget in self._attack_param_widgets:
            name = getattr(widget, "_param_name", None)
            if not name:
                continue
            value = getattr(widget, "value", None)
            # If we attached annotation/default metadata, convert strings to proper types
            annotation = getattr(widget, "_param_annotation", None)
            default = getattr(widget, "_param_default", inspect._empty)
            converted = self._convert_widget_value(value, annotation, default)
            params[name] = converted
        return params

    def _discover_attacks(self, category: str) -> Dict[str, AttackInfo]:
        if category in self._attacks:
            return self._attacks[category]

        module_name = CATEGORY_PACKAGES[category]
        module = importlib.import_module(module_name)
        attack_infos: Dict[str, AttackInfo] = {}

        exported_names = getattr(module, "__all__", None)
        if not exported_names:
            exported_names = [name for name, obj in inspect.getmembers(
                module, inspect.isclass)]

        for name in exported_names:
            cls = getattr(module, name, None)
            if not inspect.isclass(cls):
                continue
            label = self._humanize_name(name)
            attack_infos[label] = AttackInfo(label=label, cls=cls)

        self._attacks[category] = attack_infos
        return attack_infos

    def _create_widgets_for_attack(self, attack_cls: type) -> List[widgets.Widget]:
        signature = inspect.signature(attack_cls.__init__)
        widgets_list: List[widgets.Widget] = []
        for name, parameter in signature.parameters.items():
            # Skip internals and model/preprocess-related params (remove preprocessing_fn)
            if name in {"self", "model", "preprocess", "preprocessing_fn", "preprocess_fn", "pred_fn"}:
                continue
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            widget = self._widget_for_parameter(name, parameter)
            if widget is None:
                continue
            # Attach metadata to widget so we can parse/convert at collection time
            widget._param_name = name
            widget._param_annotation = parameter.annotation
            widget._param_default = parameter.default
            widgets_list.append(widget)
        return widgets_list

    # ------------------------------------------------------------------
    # Value conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _try_number_cast(s: str):
        """Try to cast a string to int or float; fallback to original string."""
        s = s.strip()
        if not s:
            return s
        # Try int
        try:
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s)
        except Exception:
            pass
        # Try float
        try:
            return float(s)
        except Exception:
            return s

    @staticmethod
    def _convert_widget_value(value: Any, annotation: Any, default: Any) -> Any:
        """Convert widget.value (often str) into the annotated/expected Python type.

        Strategy:
        - If value is already not a str (e.g., IntText, FloatText, Checkbox), return it.
        - If value is None or empty string:
            - If default is inspect._empty and annotation allows None, return None.
            - If default provided, return default.
        - If value is a str: first attempt ast.literal_eval to parse tuples/lists/numbers/booleans.
          If that fails, fall back to comma-split for sequences, or numeric cast for scalars.
        """
        # Quick pass: already correct type
        if value is None:
            # Interpret empty/None depending on default/annotation
            if default is not inspect._empty:
                return default
            # If optional in annotation, return None
            origin = get_origin(annotation)
            if origin is not None and type(None) in get_args(annotation):
                return None
            return None

        if not isinstance(value, str):
            return value

        s = value.strip()
        if s == "":
            if default is not inspect._empty:
                return default
            origin = get_origin(annotation)
            if origin is not None and type(None) in get_args(annotation):
                return None
            return None

        # If user typed something that is obviously None-ish
        if s.lower() in {"none", "null", "nil"}:
            return None

        # Try safe literal eval first (handles tuples, lists, numbers, dicts)
        try:
            lit = ast.literal_eval(s)
            return lit
        except Exception:
            pass

        # Inspect annotation for sequence-like hints
        origin = get_origin(annotation)
        if origin in {list, tuple, set} or (isinstance(origin, type) and issubclass(origin, IterableABC)):
            # comma separated
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            # Try to cast elements to numbers where appropriate
            args = get_args(annotation)
            elem_type = None
            if args:
                # if Optional[...,] or Union, pick first non-None
                unwrapped = [a for a in args if a is not type(None)]
                if unwrapped:
                    elem_type = unwrapped[0]
            converted = []
            for p in parts:
                # If elem_type is a basic numeric type, cast accordingly
                if elem_type in {int, float, str, bool}:
                    try:
                        if elem_type is int:
                            converted.append(int(float(p)))
                        elif elem_type is float:
                            converted.append(float(p))
                        elif elem_type is bool:
                            converted.append(p.lower() in {"true", "1", "yes"})
                        else:
                            converted.append(p)
                    except Exception:
                        converted.append(AttackSelectorUI._try_number_cast(p))
                else:
                    converted.append(AttackSelectorUI._try_number_cast(p))
            if origin is tuple:
                return tuple(converted)
            if origin is set:
                return set(converted)
            return converted

        # If annotation is Literal, try to match one of the literal choices
        if get_origin(annotation) is Literal:
            literals = list(get_args(annotation))
            # try direct match
            for lit in literals:
                try:
                    if isinstance(lit, str) and lit == s:
                        return lit
                    # try numeric/string conversions
                    if isinstance(lit, (int, float)) and str(lit) == s:
                        return lit
                    if isinstance(lit, bool) and s.lower() in {"true", "false"}:
                        return s.lower() == "true"
                except Exception:
                    continue
            # fallback to literal_eval or string
            try:
                return ast.literal_eval(s)
            except Exception:
                return s

        # For scalars: try numeric cast
        num = AttackSelectorUI._try_number_cast(s)
        if isinstance(num, (int, float)):
            return num

        # Booleans
        if s.lower() in {"true", "false", "yes", "no", "1", "0"}:
            return s.lower() in {"true", "yes", "1"}

        # As a last resort return the original string
        return s

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _humanize_name(name: str) -> str:
        spaced = []
        previous_lower = False
        for ch in name:
            if ch.isupper() and previous_lower:
                spaced.append(" ")
            spaced.append(ch)
            previous_lower = ch.islower()
        human = "".join(spaced)
        return human.replace("Attack", "").strip()

    @staticmethod
    def _build_options(values: Iterable[str], empty_label: str) -> List[Tuple[str, Optional[str]]]:
        items = sorted(set(filter(None, values)))
        if not items:
            return [(empty_label, None)]
        options = [("Select...", None)]
        options.extend([(item, item) for item in items])
        return options

    @staticmethod
    def _list_directory_names(path: Optional[Path]) -> List[str]:
        if not path or not path.exists():
            return []
        return [child.name for child in path.iterdir() if child.is_dir()]

    @staticmethod
    def _list_file_names(path: Optional[Path]) -> List[str]:
        if not path or not path.exists():
            return []
        return [child.name for child in path.iterdir() if child.is_file()]

    @staticmethod
    def _first_existing(*candidates: Path) -> Optional[Path]:
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        return candidates[0] if candidates else None

    @staticmethod
    def _widget_for_parameter(name: str, parameter: inspect.Parameter) -> Optional[widgets.Widget]:
        # Unified style/layout to avoid cut-off labels
        common_style = {"description_width": "initial"}
        indent_layout = Layout(margin="0 0 0 220px")

        annotation = parameter.annotation
        default = parameter.default
        description = name.replace("_", " ").title()

        origin = get_origin(annotation)
        if origin is not None:
            if origin in {list, tuple, set} or (
                isinstance(origin, type) and issubclass(origin, IterableABC)
            ):
                return widgets.Text(description=description, placeholder="Comma separated list", style=common_style, layout=indent_layout)
            if origin is Literal:
                literals = list(get_args(annotation))
                options = literals if literals else [default]
                options = [opt for opt in options if opt is not inspect._empty]
                return widgets.Dropdown(
                    description=description,
                    options=options,
                    value=default if default in options else (
                        options[0] if options else None),
                    style=common_style,
                    layout=indent_layout,
                )
            args = get_args(annotation)
            if args:
                unwrapped = [arg for arg in args if arg is not type(None)]  # noqa: E721
                if len(unwrapped) == 1:
                    annotation = unwrapped[0]

        if default is inspect._empty:
            return widgets.Text(description=description, placeholder="Required", style=common_style, layout=indent_layout)

        if isinstance(default, bool):
            return widgets.Checkbox(description=description, value=default, style=common_style, layout=indent_layout)
        if isinstance(default, int):
            return widgets.IntText(description=description, value=default, style=common_style, layout=indent_layout)
        if isinstance(default, float):
            return widgets.FloatText(description=description, value=default, style=common_style, layout=indent_layout)
        if isinstance(default, str):
            return widgets.Text(description=description, value=default, style=common_style, layout=indent_layout)
        if default is None:
            return widgets.Text(description=description, placeholder="None", style=common_style, layout=indent_layout)

        return widgets.Text(description=description, value=str(default), style=common_style, layout=indent_layout)
