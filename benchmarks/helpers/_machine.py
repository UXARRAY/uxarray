"""Pins the machine name asv files results under.

asv defaults it to the hostname, which changes every hosted-runner job and
every cluster node, so results cannot be compared across runs or merged across
shards (``results/<machine>/...``). ``asv machine --machine NAME`` alone drops
``cpu``/``num_cpu``/``ram``, hence detect first, rename after. Idempotent.

Usage::

    asv machine --yes
    python -m benchmarks.helpers._machine --name gh-Linux-X64
"""

import argparse
import json
import os
import platform
import re
import sys
from pathlib import Path

__all__ = ["pin"]

_VERSION_KEY = "version"


def default_name():
    """Returns ``(name, source_variable)`` stable across nodes of one cluster.

    ``$ASV_MACHINE``, then ``$NCAR_HOST`` (names the cluster, not the node),
    then the node name minus trailing digits. Login and compute nodes do not
    share a stem, so set ``ASV_MACHINE`` if you use both without ``NCAR_HOST``.
    """
    for variable in ("ASV_MACHINE", "NCAR_HOST"):
        value = os.environ.get(variable)
        if value:
            return value, variable
    node = platform.node().split(".")[0]
    return re.sub(r"[-_]?\d+$", "", node) or node, None


def default_path():
    """asv's machine file (``MachineCollection.get_machine_file_path``)."""
    return Path.home() / ".asv-machine.json"


def pin(name, path=None, hostname=None, sole=False):
    """Renames the machine file's freshly detected entry to ``name``; returns it.

    The fresh entry is the one keyed by this hostname, else one already named
    ``name``, else a lone entry. Other entries are kept unless ``sole``.
    """
    path = Path(path) if path is not None else default_path()
    hostname = hostname if hostname is not None else platform.node()
    stored = json.loads(path.read_text())
    version = stored.pop(_VERSION_KEY, None)

    if hostname in stored:
        detected = stored.pop(hostname)
    elif name in stored:
        detected = stored[name]
    elif len(stored) == 1:
        (only,) = stored
        detected = stored.pop(only)
    else:
        raise ValueError(
            f"{path} holds {len(stored)} machines ({', '.join(sorted(stored))}), none of "
            f"them this host ({hostname!r}) and none of them {name!r}; cannot tell which "
            f"describes the machine this is running on. Run ``asv machine --yes`` first, "
            f"or pass --name one of the recorded machines"
        )

    detected["machine"] = name
    if sole:
        stored = {}
    stored[name] = detected
    if version is not None:
        stored[_VERSION_KEY] = version
    path.write_text(json.dumps(stored, indent=4))
    return detected


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.helpers._machine",
        description="Rename asv's detected machine entry to a fixed name.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Default $ASV_MACHINE, $NCAR_HOST, then hostname minus trailing digits.",
    )
    parser.add_argument("--path", default=None, help="Machine file (default ~/.asv-machine.json).")
    parser.add_argument(
        "--hostname", default=None, help="Host whose entry to rename (default this one)."
    )
    parser.add_argument(
        "--sole",
        action="store_true",
        help="Drop other machines, so bare asv run/show work on any node without -m.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Print only the pinned name, for capturing."
    )
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print the name that would be pinned and change nothing.",
    )
    args = parser.parse_args(argv)

    name, source = (args.name, "--name") if args.name else default_name()
    if args.print_only:
        print(name)
        return 0
    detected = pin(name, args.path, args.hostname, sole=args.sole)
    if args.quiet:
        print(name)
        return 0
    print(
        f"{name}: {detected.get('cpu', '?')} "
        f"({detected.get('num_cpu', '?')} cpu, {detected.get('os', '?')})"
    )
    if source is None:
        print(
            f"  note: {name!r} came from this host's name. A cluster's login and compute "
            f"nodes do not share a stem, so set ASV_MACHINE (or rely on NCAR_HOST) if you "
            f"benchmark from both, or the results will still split in two.",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
