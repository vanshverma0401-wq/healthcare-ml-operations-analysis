#!/usr/bin/env python3
"""Export reviewed academic evidence without running original notebook code.

Added in November 2026 for portfolio publication. The input is an authorized,
privately retained source directory; no archive, dataset or access URL is copied.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import re
import struct

VERSION = "2026-11-08"
PREVIEW = re.compile(r"Sample rows after preparation:\n.*?\[5 rows x 30 columns\]", re.S)
WINDOWS_INPUT = re.compile(r'r?"[A-Za-z]:\\Users\\[^"\n]*?\.csv"')
PRIVATE = re.compile(r"/(?:Users|home)/|[A-Za-z]:\\Users\\|sharepoint\.com|instructure\.com|"
                     r"\bgh[pousr]_[A-Za-z0-9]{20,}|\bsk-proj-[A-Za-z0-9_-]{20,}", re.I)
ACTIVE = re.compile(r"<\s*(?:script|iframe|object|embed)\b|javascript\s*:", re.I)
NOTE = "[Publication redaction: five synthetic record-level preview rows omitted; all aggregate results retained.]"


class ExportError(ValueError):
    pass


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "".join(value)
    raise ExportError("Expected a string or list of strings")


def clean_text(value: str) -> tuple[str, list[str]]:
    changes = []
    value, count = PREVIEW.subn(NOTE, value)
    if count:
        changes.append("Five-row synthetic record preview replaced by an explicit publication redaction note.")
    if "/content/presentation_visuals" in value:
        value = value.replace("/content/presentation_visuals", "presentation_visuals")
        changes.append("Hosted-notebook output directory normalized to a relative presentation_visuals path.")
    if PRIVATE.search(value) or ACTIVE.search(value):
        raise ExportError("Unreviewed private or active text; manual review is required")
    return value, changes


def png_dimensions(value: bytes) -> tuple[int, int]:
    if len(value) < 24 or value[:8] != b"\x89PNG\r\n\x1a\n":
        raise ExportError("Saved image is not a PNG")
    return struct.unpack(">II", value[16:24])


def export_notebook(raw: bytes, key: str, public_path: str, root: Path) -> tuple[dict, list[dict]]:
    if len(raw) > 16 * 1024 * 1024:
        raise ExportError("Notebook exceeds the review limit")
    original = json.loads(raw)
    if not isinstance(original.get("cells"), list):
        raise ExportError("Missing notebook cells")
    public = copy.deepcopy(original)
    public["nbformat"] = 4
    public["nbformat_minor"] = 5
    public["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                          "language_info": {"name": "python"}}
    mappings, figures = [], []
    figure_number = 0
    last_heading = ""
    for index, cell in enumerate(public["cells"]):
        source = text(cell.get("source", []))
        cleaned, changes = clean_text(source)
        cell["source"] = cleaned.splitlines(keepends=True)
        cell["metadata"] = {}
        cell["id"] = f"{key}-cell-{index}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
        cell_map = {"source_cell_index": index, "cell_type": cell["cell_type"],
                    "original_source_sha256": sha(source.encode()), "public_source_sha256": sha(cleaned.encode()),
                    "source_changed": source != cleaned, "changes": changes, "outputs": []}
        outputs = []
        for oi, output in enumerate(cell.get("outputs", [])):
            kind = output.get("output_type")
            out_map = {"source_output_index": oi, "output_type": kind, "changes": []}
            if kind == "stream":
                value = text(output.get("text", []))
                if key == "v5" and index == 9:
                    value = "[Publication note: original package-installation log omitted; installation command retained above.]\n"
                    out_map["changes"].append("Package-installation log replaced by an explicit note; source command retained.")
                else:
                    value, out_map["changes"] = clean_text(value)
                outputs.append({"output_type": "stream", "name": output.get("name", "stdout"), "text": value.splitlines(keepends=True)})
            elif kind in {"display_data", "execute_result"}:
                source_data = output.get("data", {})
                if not isinstance(source_data, dict):
                    raise ExportError("Invalid output data")
                unsupported = set(source_data) - {"image/png", "text/plain", "text/markdown", "text/html"}
                if unsupported:
                    raise ExportError("Unknown output MIME types need manual review")
                data = {}
                for mime in ["text/plain", "text/markdown"]:
                    if mime in source_data:
                        value, edits = clean_text(text(source_data[mime]))
                        data[mime] = value.splitlines(keepends=True)
                        out_map["changes"].extend(edits)
                        if mime == "text/markdown" and value.startswith("## "):
                            last_heading = value.strip().lstrip("# ")
                if "text/html" in source_data:
                    if not data:
                        raise ExportError("An HTML-only output needs manual review before publication")
                    out_map["changes"].append("HTML representation removed; original plain-text aggregate table retained.")
                if "image/png" in source_data:
                    image = base64.b64decode("".join(text(source_data["image/png"]).split()), validate=True)
                    width, height = png_dimensions(image)
                    figure_number += 1
                    asset = f"assets/billing/{key}-{figure_number:02d}.png"
                    destination = root / asset
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(image)
                    data["image/png"] = source_data["image/png"]
                    title = last_heading if key == "v4" else {
                        5: "Billing-category confusion matrix",
                        6: "Length of stay by synthetic billing category",
                        7: "Original billing feature importance",
                        8: "Billing-category distribution",
                    }.get(index, f"Original V5 figure {figure_number}")
                    figures.append({"workstream": key, "source_notebook_sha256": sha(raw),
                                    "source_cell_index": index, "source_output_index": oi,
                                    "asset": asset, "title": title, "sha256": sha(image),
                                    "width": width, "height": height,
                                    "saved_original_bytes_preserved": True, "freshly_recomputed": False})
                if not data:
                    raise ExportError("An output would be silently lost")
                out = {"output_type": kind, "data": data, "metadata": {}}
                if kind == "execute_result":
                    out["execution_count"] = None
                outputs.append(out)
            else:
                raise ExportError("Unsupported output type needs manual review")
            cell_map["outputs"].append(out_map)
        if cell["cell_type"] == "code":
            cell["outputs"] = outputs
        mappings.append(cell_map)
    destination = root / public_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(public, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    result = {"workstream": key, "original_basename": "final.ipynb" if key == "v4" else "V5.ipynb",
              "original_sha256": sha(raw), "public_path": public_path,
              "public_sha256": sha(destination.read_bytes()), "source_cell_count": len(original["cells"]),
              "public_cell_count": len(public["cells"]),
              "source_code_cells": sum(c["cell_type"] == "code" for c in original["cells"]),
              "png_count": len(figures), "original_outputs_count": sum(len(c.get("outputs", [])) for c in original["cells"]),
              "public_outputs_count": sum(len(c.get("outputs", [])) for c in public["cells"]),
              "personal_metadata_removed": True, "public_cell_ids_generated": True, "execution_counts_cleared": True,
              "cell_mapping": mappings, "notebook_execution_performed": False}
    return result, figures


def export_python(source: Path, public_path: str, root: Path) -> dict:
    raw = source.read_bytes()
    original = raw.decode("utf-8-sig")
    cleaned, count = WINDOWS_INPUT.subn('"data/healthcare_data.csv"', original)
    if PRIVATE.search(cleaned):
        raise ExportError("Unreviewed private source path or credential")
    ast.parse(cleaned)
    destination = root / public_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(cleaned, encoding="utf-8")
    old_lines, new_lines = original.splitlines(), cleaned.splitlines()
    changed = [i + 1 for i, (old, new) in enumerate(zip(old_lines, new_lines)) if old != new]
    if len(old_lines) != len(new_lines):
        raise ExportError("Source line count changed")
    return {"original_basename": source.name, "original_sha256": sha(raw), "public_path": public_path,
            "public_sha256": sha(destination.read_bytes()), "source_lines": len(old_lines),
            "private_csv_literals_replaced": count, "changed_source_lines": changed,
            "transformation": "Private absolute input CSV path normalized to a relative path." if count else "Original text retained; UTF-8 encoding normalized only.",
            "original_code_executed": False, "syntax_parse_passed": True}


def fenced(value: str, language: str = "") -> str:
    fence = "`" * max(3, max((len(m.group()) + 1 for m in re.finditer(r"`+", value)), default=3))
    return f"{fence}{language}\n{value}" + ("" if value.endswith("\n") else "\n") + fence + "\n"


def render_report(root: Path, notebooks: list[dict], figures: list[dict]) -> dict:
    parts = ["# Complete billing notebook evidence\n",
             "Original team code, narrative and privacy-safe saved outputs from the final V4 and V5 notebook artifacts. All source cells are represented in order. This is a historical academic project using synthetic records, not a new model run.\n",
             "The five-row record preview is explicitly redacted, private environment paths are normalized, the package-installation log is marked as omitted, and the HTML aggregate table is shown as its original plain text. Model logic, original conclusions and all 16 saved figures remain. Review [results](RESULTS.md), [methods and limitations](METHODS_AND_LIMITATIONS.md), [the figure gallery](FIGURE_GALLERY.md) and [source transformations](technical/notebook_inventory.json).\n"]
    mapping = {(f["workstream"], f["source_cell_index"], f["source_output_index"]): f for f in figures}
    cells = []
    for entry in notebooks:
        key = entry["workstream"]
        nb = json.loads((root / entry["public_path"]).read_text())
        parts.append(f"## {key.upper()} final notebook\n\n[Open the source notebook]({entry['public_path']})\n")
        for index, cell in enumerate(nb["cells"]):
            source = text(cell["source"])
            parts.append(f'<a id="{key}-cell-{index}"></a>\n')
            if cell["cell_type"] == "markdown":
                # Cell 1 in the V4 source is a literal saved text transcript.
                parts.append(fenced(source) if key == "v4" and index == 1 else source + "\n")
            else:
                parts += [f"<details>\n<summary>Original {key.upper()} source cell {index + 1}</summary>\n", fenced(source, "python"), "</details>\n"]
                for oi, output in enumerate(cell.get("outputs", [])):
                    if output["output_type"] == "stream":
                        parts += ["Saved text output:\n", fenced(text(output["text"]))]
                    else:
                        data = output["data"]
                        if "image/png" in data:
                            f = mapping[(key, index, oi)]
                            parts += [f"![{f['title']}]({f['asset']})\n", "*Original saved figure; synthetic educational data. No fresh execution.*\n"]
                        if "text/markdown" in data:
                            parts.append(text(data["text/markdown"]) + "\n")
                        elif "text/plain" in data and "image/png" not in data:
                            parts.append(fenced(text(data["text/plain"])))
            cells.append({"workstream": key, "source_cell_index": index, "source_text_sha256": sha(source.encode()),
                          "report_anchor": f"{key}-cell-{index}"})
    content = "\n".join(parts).rstrip() + "\n"
    (root / "FULL_BILLING_REPORT.md").write_text(content, encoding="utf-8")
    return {"report": "FULL_BILLING_REPORT.md", "sha256": sha(content.encode()), "rendered_cells": len(cells),
            "saved_pngs": len(figures), "notebook_execution_performed": False, "cells": cells}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source, root = args.source_root.resolve(), args.output_root.resolve()
    if source == root or source in root.parents or root in source.parents:
        parser.error("Private source and public output directories must be separate")
    root.mkdir(parents=True, exist_ok=True)
    notebook_specs = [
        ("v4", "Billing Amounts Work Final/Billing Amounts Work/final.ipynb", "notebooks/billing_v4_final.ipynb"),
        ("v5", "Billing Work Final/Billing Work Final/V5.ipynb", "notebooks/billing_v5_extension.ipynb"),
    ]
    notebooks, figures = [], []
    for key, src, dest in notebook_specs:
        item, images = export_notebook((source / src).read_bytes(), key, dest, root)
        notebooks.append(item)
        figures.extend(images)
    py_specs = [("Billing Amounts Work Final/Billing Amounts Work/FINAL/Predict_High_Cases.py", "src/billing/predict_high_cost_cases.py")]
    katie = source / "Billing Amounts Work Final/Billing Amounts Work/Katie_s Work"
    for path in sorted(katie.glob("*.py")):
        normalized = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_") + ".py"
        py_specs.append((str(path.relative_to(source)), "src/billing/" + normalized))
    op_base = source / "Healthcare Operations Risk Management_Predicting Staffed Capacity/Healthcare Operations Risk Management_Predicting Staffed Capacity/Drew Fleming"
    for label, folder in [("baseline", "Baseline Model/Healthcare Operations Risk Management_Baseline Model"),
                          ("refined", "Refined Model/Healthcare Operations Risk Management_Refined Model")]:
        match = list((op_base / folder).glob("*.py"))
        if len(match) != 1:
            raise ExportError("Ambiguous operations source")
        py_specs.append((str(match[0].relative_to(source)), f"src/operations/{label}.py"))
    scripts = [export_python(source / src, dest, root) for src, dest in py_specs]
    report = render_report(root, notebooks, figures)
    technical = root / "technical"
    technical.mkdir(exist_ok=True)
    for filename, content in [
        ("notebook_inventory.json", {"export_date": VERSION, "notebooks": notebooks, "report": report}),
        ("figure_manifest.json", {"export_date": VERSION, "saved_png_count": len(figures), "figures": figures}),
        ("source_manifest.json", {"export_date": VERSION, "source_context": "User-provided authenticated final-project archive", "notebooks": [{k: v for k, v in n.items() if k != "cell_mapping"} for n in notebooks], "python_sources": scripts, "original_code_executed": False}),
    ]:
        (technical / filename).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    gallery = ["# Billing figure gallery\n", "All 16 figures below are unchanged saved images from the final V4 and V5 notebooks. The data is synthetic. They have not been recomputed or clinically validated. See [results](RESULTS.md) and [methods and limitations](METHODS_AND_LIMITATIONS.md) for version and evaluation qualifications.\n", "[Complete billing notebooks and safe outputs](FULL_BILLING_REPORT.md) · [Operations figure gallery](OPERATIONS_GALLERY.md)\n"]
    for i, f in enumerate(figures, 1):
        gallery += [f"## Figure {i} {f['title']}\n", f"![{f['title']}]({f['asset']})\n", f"Original {f['workstream'].upper()} saved output, zero-based source cell {f['source_cell_index']}, output {f['source_output_index']}. Original wording is retained; feature-importance and clinical-style titles do not establish causal or clinical validity.\n"]
    (root / "FIGURE_GALLERY.md").write_text("\n".join(gallery), encoding="utf-8")
    print(json.dumps({"notebooks": len(notebooks), "source_cells": sum(n["source_cell_count"] for n in notebooks),
                      "original_pngs": len(figures), "python_sources": len(scripts), "original_code_executed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
