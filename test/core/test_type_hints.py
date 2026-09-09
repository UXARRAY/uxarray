import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _annotation(source: Path, function_name: str, parameter_name: str) -> str:
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    parameter = next(arg for arg in function.args.args if arg.arg == parameter_name)
    return ast.unparse(parameter.annotation)


def test_use_dual_type_hints_do_not_allow_none():
    api = ROOT / "uxarray" / "core" / "api.py"
    grid = ROOT / "uxarray" / "grid" / "grid.py"

    assert _annotation(api, "open_grid", "use_dual") == "bool"
    assert _annotation(api, "open_dataset", "use_dual") == "bool"
    assert _annotation(api, "open_mfdataset", "use_dual") == "bool"
    assert _annotation(grid, "from_dataset", "use_dual") == "bool"
