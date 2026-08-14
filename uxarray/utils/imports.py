"""
Purpose: utils related to imports, e.g. handling optional dependency imports
"""

import importlib

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
