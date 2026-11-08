"""Synthetic privacy and preservation checks; original models are not executed."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_public_evidence.py"
SPEC = importlib.util.spec_from_file_location("export_public_evidence", SCRIPT)
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aBqUAAAAASUVORK5CYII=")


def fixture():
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"author": "PRIVATE_AUTHOR"}, "cells": [
        {"cell_type": "markdown", "metadata": {"private_tag": "PRIVATE_TAG"},
         "source": ["Before aggregate result\n", "Sample rows after preparation:\nPRIVATE_RECORD\n[5 rows x 30 columns]\n", "After aggregate result\n"]},
        {"cell_type": "code", "execution_count": 9, "metadata": {},
         "source": ["raise RuntimeError('ORIGINAL_CODE_MUST_NOT_EXECUTE')\n"],
         "outputs": [{"output_type": "stream", "text": ["Accuracy: 0.42\n"]},
                     {"output_type": "display_data", "data": {"image/png": base64.b64encode(PNG).decode(),
                                                               "text/plain": ["<Figure>"]}, "metadata": {}}]},
    ]}


class ExportTests(unittest.TestCase):
    def test_complete_cells_code_metrics_and_png_preserved_without_execution(self):
        raw = json.dumps(fixture()).encode()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory, figures = export.export_notebook(raw, "v4", "notebooks/final.ipynb", root)
            nb = json.loads((root / "notebooks/final.ipynb").read_text())
            self.assertEqual(len(nb["cells"]), 2)
            self.assertEqual(nb["cells"][1]["source"], fixture()["cells"][1]["source"])
            self.assertEqual(nb["cells"][1]["outputs"][0]["text"], ["Accuracy: 0.42\n"])
            self.assertEqual((root / figures[0]["asset"]).read_bytes(), PNG)
            self.assertEqual(figures[0]["sha256"], hashlib.sha256(PNG).hexdigest())
            self.assertEqual(inventory["source_cell_count"], inventory["public_cell_count"])
            self.assertFalse(inventory["notebook_execution_performed"])
            public = (root / "notebooks/final.ipynb").read_text()
            for private in ["PRIVATE_AUTHOR", "PRIVATE_TAG", "PRIVATE_RECORD"]:
                self.assertNotIn(private, public)
            self.assertIn("Before aggregate result", public)
            self.assertIn("After aggregate result", public)
            self.assertIn("Publication redaction", public)

    def test_unreviewed_private_text_active_output_and_unknown_mime_are_rejected(self):
        cases = []
        for unsafe in ["/Users/private/record.csv", "https://tenant.sharepoint.com/private", "<script>doSomething()</script>"]:
            value = fixture()
            value["cells"][0]["source"] = [unsafe]
            cases.append(value)
        value = fixture()
        value["cells"][1]["outputs"] = [{"output_type": "display_data", "data": {"application/javascript": "alert(1)"}}]
        cases.append(value)
        with tempfile.TemporaryDirectory() as temp:
            for value in cases:
                with self.assertRaises(export.ExportError):
                    export.export_notebook(json.dumps(value).encode(), "v4", "out.ipynb", Path(temp))
            self.assertFalse((Path(temp) / "out.ipynb").exists())

    def test_html_table_removed_while_original_aggregate_plain_text_is_retained(self):
        value = fixture()
        value["cells"][1]["outputs"] = [{"output_type": "display_data", "data": {
            "text/html": "<table><tr><td>PRIVATE_HTML_ONLY</td></tr></table>",
            "text/plain": ["Model        Macro_F1\nRandomForest 0.41\n"]}}]
        with tempfile.TemporaryDirectory() as temp:
            inventory, _ = export.export_notebook(json.dumps(value).encode(), "v5", "out.ipynb", Path(temp))
            public = json.loads((Path(temp) / "out.ipynb").read_text())
            data = public["cells"][1]["outputs"][0]["data"]
            self.assertNotIn("text/html", data)
            self.assertEqual("".join(data["text/plain"]), "".join(value["cells"][1]["outputs"][0]["data"]["text/plain"]))
            self.assertIn("HTML representation removed", inventory["cell_mapping"][1]["outputs"][0]["changes"][0])
            value["cells"][1]["outputs"][0]["data"].pop("text/plain")
            with self.assertRaisesRegex(export.ExportError, "HTML-only"):
                export.export_notebook(json.dumps(value).encode(), "v5", "second.ipynb", Path(temp))

    def test_private_csv_path_normalization_preserves_original_logic_and_line_count(self):
        original = 'FILE_PATH = r"D:\\Users\\PRIVATE_NAME\\PRIVATE_ORG\\input.csv"\nprint(review_lock_by_horizon_df)\n'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "original.py"
            source.write_text(original)
            result = export.export_python(source, "public/src/refined.py", root)
            public = (root / "public/src/refined.py").read_text()
            self.assertEqual(public, 'FILE_PATH = "data/healthcare_data.csv"\nprint(review_lock_by_horizon_df)\n')
            self.assertEqual(result["changed_source_lines"], [1])
            self.assertEqual(len(original.splitlines()), len(public.splitlines()))
            self.assertEqual(source.read_text(), original)
            self.assertFalse(result["original_code_executed"])

    def test_report_keeps_source_fences_and_all_saved_figure_links(self):
        value = fixture()
        value["cells"][1]["source"] = ["print('```')\n"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry, figures = export.export_notebook(json.dumps(value).encode(), "v4", "notebooks/out.ipynb", root)
            report = export.render_report(root, [entry], figures)
            text = (root / "FULL_BILLING_REPORT.md").read_text()
            self.assertIn("````python", text)
            self.assertIn("print('```')", text)
            self.assertIn("v4-cell-0", text)
            self.assertIn("v4-cell-1", text)
            self.assertIn(figures[0]["asset"], text)
            self.assertEqual(report["rendered_cells"], 2)
            self.assertEqual(report["saved_pngs"], 1)


if __name__ == "__main__":
    unittest.main()
