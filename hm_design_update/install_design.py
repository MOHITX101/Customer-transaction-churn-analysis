"""Install presentation updates into the existing H&M project with a backup.

Run from the project root:
    ./.venv/Scripts/python.exe hm_design_update/install_design.py
Only application presentation files are changed. No model/data writes.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import io
import json
import re
import shutil
import tokenize
import tomllib

BUNDLE = Path(__file__).resolve().parent
PAGES = [
    "1_Sales_Analysis.py", "2_Customer_Analysis.py", "3_Customer_Predictions.py",
    "4_Product_Recommendations.py", "5_Weekly_Forecasting.py",
    "6_Model_Explanations.py", "7_Campaign_Simulator.py",
]


def is_st_call(node, name):
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "st"
        and node.func.attr == name
    )


def style_page(source, filename):
    """Remove duplicated page chrome; change only explicit chart color strings."""
    tree = ast.parse(source)
    removals = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and is_st_call(node.value, "set_page_config"):
            removals.append((node.lineno, node.end_lineno))
        elif isinstance(node, ast.With):
            is_sidebar = any(
                isinstance(item.context_expr, ast.Attribute)
                and isinstance(item.context_expr.value, ast.Name)
                and item.context_expr.value.id == "st"
                and item.context_expr.attr == "sidebar"
                for item in node.items
            )
            # Remove only the known branding block, never a sidebar containing controls.
            safe_statements = all(
                isinstance(stmt, ast.Expr) and any(
                    is_st_call(stmt.value, name)
                    for name in ["title", "caption", "divider", "write"]
                )
                for stmt in node.body
            )
            known_title = any(
                isinstance(stmt, ast.Expr) and is_st_call(stmt.value, "title")
                and stmt.value.args and isinstance(stmt.value.args[0], ast.Constant)
                and stmt.value.args[0].value == "Retail Intelligence"
                for stmt in node.body
            )
            if is_sidebar and safe_statements and known_title:
                removals.append((node.lineno, node.end_lineno))

    lines = source.splitlines(keepends=True)
    for start, end in sorted(removals, reverse=True):
        del lines[start - 1:end]
    source = "".join(lines)
    replacements = {
        "#0D9488": "#C7002C", "#2563EB": "#343434",
        "#F59E0B": "#A8794F", "#64748B": "#6F6F6F",
    }
    # Red raises risk; graphite lowers risk. Never give both the same color.
    if "Model_Explanations" in filename:
        replacements["#0D9488"] = "#4B5563"
    if "Customer_Analysis" in filename:
        replacements["#0D9488"] = "#343434"
        replacements["#F59E0B"] = "#C7002C"

    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                value = None
            if isinstance(value, str) and value in replacements:
                token = token._replace(string=repr(replacements[value]))
        tokens.append(token)
    result = tokenize.untokenize(tokens)
    compile(result, filename, "exec")
    return result


def merge_theme(original, replacement):
    """Replace [theme] only; retain server settings, secrets references, etc."""
    tomllib.loads(original)
    lines, output, skipping = original.splitlines(keepends=True), [], False
    for line in lines:
        if re.match(r"^\s*\[theme\]\s*(?:#.*)?$", line):
            skipping = True
            continue
        if re.match(r"^\s*\[", line):
            skipping = False
        if not skipping:
            output.append(line)
    merged = "".join(output).rstrip() + "\n\n" + replacement
    tomllib.loads(merged)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true", help="Preview changes without writing.")
    args = parser.parse_args()
    root = args.project_root.resolve()
    required = [root / "dashboard/app.py"] + [root / "dashboard/pages" / name for name in PAGES]
    required += [root / "images/H&M-Logo.svg.webp", root / "images/images.jpg"]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Run from the H&M project root. Missing:\n" + "\n".join(missing))

    changes = {}
    for source in sorted((BUNDLE / "payload").rglob("*")):
        if not source.is_file() or source.suffix not in {".py", ".css", ".toml"}:
            continue
        relative = source.relative_to(BUNDLE / "payload")
        target = root / relative
        text = source.read_text(encoding="utf-8")
        if relative == Path(".streamlit/config.toml") and target.exists():
            text = merge_theme(target.read_text(encoding="utf-8-sig"), text)
        if source.suffix == ".py":
            compile(text, str(relative), "exec")
        changes[target] = text
    for name in PAGES:
        target = root / "dashboard/pages" / name
        changes[target] = style_page(target.read_text(encoding="utf-8-sig"), name)

    # All validation happens before any project file is changed.
    for target in changes:
        if target.is_symlink():
            raise SystemExit(f"Refusing to overwrite a symbolic link: {target}")
        print("Update:", target.relative_to(root))
    if args.check:
        print("Preview complete. No project files changed.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup = root / "backups" / f"before_hm_design_{timestamp}"
    backup.mkdir(parents=True)
    manifest = []
    for target in changes:
        relative = target.relative_to(root)
        existed = target.exists()
        manifest.append({"path": str(relative), "existed": existed})
        if existed:
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    written = []
    try:
        for target, text in changes.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            written.append(target)
            target.write_text(text, encoding="utf-8")
    except Exception:
        for target in reversed(written):
            saved = backup / target.relative_to(root)
            if saved.exists():
                shutil.copy2(saved, target)
            elif target.exists():
                target.unlink()
        raise
    print("\nDesign update installed.")
    print("Backup:", backup)
    print("Models, datasets and images were not changed.")
    print("Restart: ./.venv/Scripts/python.exe -m streamlit run dashboard/app.py --server.address localhost")


if __name__ == "__main__":
    main()
