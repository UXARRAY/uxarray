"""
Purpose: utils related to imports, e.g. handling optional dependency imports
"""

import ast
import importlib
from dataclasses import dataclass, field
from pathlib import Path

from uxarray.errors import OptionalDependencyNotFoundError

# Mapping from optional dependency to corresponding extras, to help improve
# error messages in case of forgetting to install necessary optional deps.
# Hard-coded intentionally to avoid extra overhead of looking up package info,
# and to avoid any confusion if installed from wheels.
# Intentionally excluded "dev" because "dev" is mostly for tests.
_OPTIONAL_DEPS_TO_EXTRAS = {
    "cartopy": ("geo", "viz"),
    "geopandas": "geo",
    "healpix": "geo",
    "pyproj": "geo",
    "spatialpandas": "geo",
    "datashader": "viz",
    "matplotlib": "viz",
    "geoviews": "viz",
    "holoviews": "viz",
    "hvplot": "viz",
}

# Name of the helper that raises a hint when optional deps are missing.
_HINT_FUNC_NAME = "_raise_hint_if_optional_deps_missing"


def _raise_hint_if_optional_deps_missing(*packages: str):
    """try to import these optional dependencies; raise helpful hint if any ModuleNotFoundError.

    packages: str
        names of packages to try to import.
        Must be keys in _OPTIONAL_DEPS_TO_EXTRAS, i.e. one or more of the following:
            cartopy, geopandas, healpix, pyproj, spatialpandas,
            datashader, matplotlib, geoviews, holoviews, hvplot
    """
    # whitelist package names; crash if anything unexpected is provided.
    _unknown = [pkg for pkg in packages if pkg not in _OPTIONAL_DEPS_TO_EXTRAS]
    if len(_unknown) > 0:
        raise ValueError(
            f"Unrecognized package names in _raise_hint_if_optional_deps_missing(): {_unknown}. "
            f"Recognized names are: {list(_OPTIONAL_DEPS_TO_EXTRAS.keys())}"
        )  # (if this error occurs, it is almost certainly a bug in the UXarray codebase itself.)

    # note: want to provide one error covering all missing modules, to improve user experience.
    # also note: if cartopy is one of the modules, need to be smart about error message.
    missing = []
    last_err = None
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ModuleNotFoundError as err:
            missing.append(pkg)
            last_err = err  # will raise result from last_err to keep some error traceback info.

    if len(missing) == 0:
        pass  # nothing to do; all requested packages imported successfully!
    else:  # raise error with helpful message.
        # Trying to be slightly smart with the message here, to be improve user experience:
        # (1) if everything would be covered by one extra, suggest it. (E.g. holoviews & cartopy --> [viz])
        # (2) if everything would easily be covered by doing multiple extras, suggest them,
        #   and also mention [all] as an option. (E.g. healpix & holoviews --> [geo,viz] or [all])
        # (3) if just one missing package, with multiple extras, suggest "or" (E.g. just cartopy --> [geo] or [viz])
        # (4) in any other case, stop trying to be smart; just suggest [all].
        need_extras = set()
        or_extras = []
        missing_extras = {pkg: _OPTIONAL_DEPS_TO_EXTRAS[pkg] for pkg in missing}
        one_extra = {
            pkg: extra
            for pkg, extra in missing_extras.items()
            if isinstance(extra, str)
        }
        many_extras = {
            pkg: extras
            for pkg, extras in missing_extras.items()
            if not isinstance(extras, str)
        }
        assert all(
            len(extras) >= 2 for extras in many_extras.values()
        )  # else wrong format in _OPTIONAL_DEPS_TO_EXTRAS.
        for pkg, extra in one_extra.items():
            need_extras.add(extra)  # definitely need to include all of these
        for pkg, extras in many_extras.items():
            if any(
                extra in need_extras for extra in extras
            ):  # still maybe in case (1) or (2).
                pass  # this package is already covered by other needed extras!
            elif len(many_extras) == 1:  # case (3)
                need_extras.add(extras[0])
                or_extras.extend(
                    extra for extra in extras[1:] if extra not in or_extras
                )
                if "all" not in or_extras:
                    or_extras.append("all")
            else:  # case (4)
                need_extras = set(["all"])
                break
        need_extras_str = ",".join(sorted(need_extras))  # sort is just for style
        errmsg = "Failed to import: " + ", ".join(sorted(missing))
        errmsg += f'.\nConsider running ``pip install "uxarray[{need_extras_str}]"``'
        if len(need_extras) >= 2:
            errmsg += ' or ``pip install "uxarray[all]"``'
        elif len(or_extras) == 1:
            errmsg += f' or ``pip install "uxarray[{or_extras[0]}]"``'
        elif len(or_extras) == 2:
            errmsg += f', ``pip install "uxarray[{or_extras[0]}]"``, or ``pip install "uxarray[{or_extras[1]}]"``'
        elif len(or_extras) >= 3:
            sorted_or_extras = sorted(
                or_extras, key=lambda s: (s == "all", s)
            )  # put "all" last
            options_str = ", ".join(f"[{extra}]" for extra in sorted_or_extras[:-1])
            options_str += f", or [{sorted_or_extras[-1]}]"
            errmsg += f" or pip install with {options_str}"
        errmsg += ", then try again."
        raise OptionalDependencyNotFoundError(errmsg) from last_err


# ------- help methods to check for _raise_hint_... calls ------- #
# (used by pytest test suite to ensure that all optional-dependency imports
# are properly guarded by a call to _raise_hint_if_optional_deps_missing().)

@dataclass
class OptionalImportCheckResult:
    """Result of checking a function for optional import usage; contains:
        qualname: str
            dotted qualname of the function/method,
            e.g. "method1" or "MyClass.method2" or "outer_func.inner_func"
        filepath: Path
            path to the source file containing the function/method
        lineno: int
            line number of the function/method definition in the source file
        imported_deps: set
            all deps from _OPTIONAL_DEPS_TO_EXTRAS that were
            imported directly in the function
        hinted_deps: set
            all deps from _OPTIONAL_DEPS_TO_EXTRAS that were passed
            to _raise_hint_if_optional_deps_missing() in the function
        """
    qualname: str
    filepath: Path
    lineno: int
    imported_deps: set = field(default_factory=set)
    hinted_deps: set = field(default_factory=set)

    @property
    def missing_deps(self) -> set:
        """Deps imported but not covered by the hint call. Should be empty."""
        return self.imported_deps - self.hinted_deps

    @property
    def extra_deps(self) -> set:
        """Deps hinted but never actually imported. Worth a warning, not fatal."""
        return self.hinted_deps - self.imported_deps


def _top_level_module_from_dotted_name(dotted_name: str) -> str:
    """returns top-level module name, from full dotted name.
    E.g. "cartopy.crs" --> "cartopy", "matplotlib.pyplot" --> "matplotlib"
    """
    return dotted_name.split(".")[0]


def _find_package_source_files(root: Path):
    """
    Yield .py files under `root` that live in a real package, i.e. whose
    immediate parent directory also contains an __init__.py. Directories
    without an __init__.py are treated as scripts and skipped.
    """
    for path in sorted(root.rglob("*.py")):
        if (path.parent / "__init__.py").exists():
            yield path


def _analyze_optional_imports_in_function(
        func_node,
        filepath: Path,
        qualname: str
    ) -> OptionalImportCheckResult:
    """Returns OptionalImportCheckResult for a function,
    telling which optional dependencies were imported and which were hinted
    via _raise_hint_if_optional_deps_missing().

    Inspects only the *direct* statements of a function body (not nested
    blocks, not nested functions). _raise_hint_if_optional_deps_missing()
    should be called directly in all functions with optional imports.
    """
    result = OptionalImportCheckResult(qualname=qualname, filepath=filepath, lineno=func_node.lineno)

    for stmt in func_node.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                top = _top_level_module_from_dotted_name(alias.name)
                if top in _OPTIONAL_DEPS_TO_EXTRAS:
                    result.imported_deps.add(top)

        elif isinstance(stmt, ast.ImportFrom):
            # skip relative imports (`from . import x`) -- not external deps
            if stmt.module and stmt.level == 0:
                top = _top_level_module_from_dotted_name(stmt.module)
                if top in _OPTIONAL_DEPS_TO_EXTRAS:
                    result.imported_deps.add(top)

        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            func = call.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr

            if name == _HINT_FUNC_NAME:
                for arg in call.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        result.hinted_deps.add(arg.value)

    return result


def _iter_functions_optional_import_checks(tree: ast.AST, filepath: Path):
    """
    Walk the module, descending through classes and nested functions while
    tracking a dotted qualname, yielding one OptionalImportCheckResult per
    function/method encountered. Assumes for now that optional imports are
    not used inside lambdas; do not visit any lambdas here.
    """
    def walk(node, scope_parts):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join(scope_parts + [child.name])
                yield _analyze_optional_imports_in_function(child, filepath, qualname)
                yield from walk(child, scope_parts + [child.name])
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, scope_parts + [child.name])
            else:
                yield from walk(child, scope_parts)

    yield from walk(tree, [])


def _optional_import_usage_throughout(src_root: str | Path) -> list[OptionalImportCheckResult]:
    """Returns list of OptionalImportCheckResult for all functions/methods in the package
    source files under `src_root` which either imports a known optional dependency directly,
    or calls the _raise_hint_if_optional_deps_missing() function, or both.
    """
    if not isinstance(src_root, Path):
        src_root = Path(src_root)
    results = []
    for filepath in _find_package_source_files(src_root):
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
        for check in _iter_functions_optional_import_checks(tree, filepath):
            if check.imported_deps or check.hinted_deps:
                results.append(check)
    return results
