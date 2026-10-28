import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[2] / "uxarray"

# None is a real value for these even though the default isn't None.
ALLOWED = {
    ("integrate", "quadrature_rule"),
    ("integrate", "order"),
    ("polygons", "xlabel"),
    ("polygons", "ylabel"),
    ("nearest_neighbor", "yac_method"),
    ("bilinear", "yac_method"),
}


def _allows_none(annotation):
    for node in ast.walk(annotation):
        if isinstance(node, ast.Constant) and node.value is None:
            return True
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "Optional":
            return True
    return False


def _none_hints_with_other_defaults():
    for path in sorted(PACKAGE.rglob("*.py")):
        for func in ast.walk(ast.parse(path.read_text())):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = func.args
            positional = args.posonlyargs + args.args
            pairs = list(zip(positional[-len(args.defaults) :], args.defaults))
            pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d]
            for arg, default in pairs:
                if arg.annotation is None or (func.name, arg.arg) in ALLOWED:
                    continue
                if isinstance(default, ast.Constant) and default.value is None:
                    continue
                if _allows_none(arg.annotation):
                    rel = path.relative_to(PACKAGE.parent)
                    yield f"{rel}:{arg.lineno} {func.name}({arg.arg})"


def test_none_only_hinted_when_handled():
    # Issue #1593: a hint shouldn't allow None unless None means something.
    # If it does, default to None or add the parameter to ALLOWED.
    offenders = list(_none_hints_with_other_defaults())
    if offenders:
        pytest.fail("Hint allows None but default isn't None:\n" + "\n".join(offenders))
