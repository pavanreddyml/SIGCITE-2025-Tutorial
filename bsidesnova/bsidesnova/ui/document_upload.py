import os
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display

class DocumentUploadWidget:
    """
    Robust FileUpload handler for ipywidgets v7 and v8.

    Fixes:
    - Works with both dict-based (v7) and tuple-based (v8) `value`.
    - Avoids deprecated 'metadata' access; uses standard 'name' and 'content'.
    - Clears and updates output reliably.
    """
    def __init__(self, save_dir: str = "."):
        self.uploaded_file_path = None
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.upload_widget = widgets.FileUpload(
            accept="",           # accept all
            multiple=False,
            description="Upload Document"
        )
        self.output = widgets.Output()
        self.upload_widget.observe(self._on_upload, names="value")

    def _extract_first_file(self, value):
        """
        Normalize FileUpload.value to (name, content) for v7/v8.

        v7: dict-like -> {'<id>': {'name': str, 'type': str, 'size': int, 'content': bytes}}
        v8: tuple of UploadedFile -> each has .name, .type, .size, .content
        """
        if not value:
            return None, None

        # v8: tuple/list of UploadedFile
        if isinstance(value, (tuple, list)) and len(value) > 0:
            f = value[0]
            # UploadedFile in v8 has attributes
            name = getattr(f, "name", None)
            content = getattr(f, "content", None)
            return name, content

        # v7: dict-like with values as dicts
        if isinstance(value, dict) and len(value) > 0:
            f = next(iter(value.values()))
            # Prefer standard keys; fall back to legacy metadata if present
            name = f.get("name") or (f.get("metadata") or {}).get("name")
            content = f.get("content")
            return name, content

        return None, None

    def _on_upload(self, change):
        name, content = self._extract_first_file(change.get("new"))
        if not name or content is None:
            with self.output:
                self.output.clear_output(wait=True)
                print("⚠️ No file detected. Try again.")
            return

        # Sanitize filename and save
        safe_name = os.path.basename(name)
        path = self.save_dir / safe_name
        with open(path, "wb") as f:
            f.write(content)

        self.uploaded_file_path = str(path)

        with self.output:
            self.output.clear_output(wait=True)
            print(f"✅ File uploaded successfully: {self.uploaded_file_path}")

    def display(self):
        display(self.upload_widget, self.output)

    def get_file_path(self):
        return self.uploaded_file_path
