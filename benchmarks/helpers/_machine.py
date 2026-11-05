"""Records this host in asv's machine file under a stable name, and prints it.

asv names the machine after the hostname, which changes every hosted-runner
job and every cluster node, so results cannot be compared across runs or
merged across shards (``results/<machine>/...``). ``asv machine --machine
NAME`` alone drops the detected ``cpu``/``num_cpu``/``ram``, hence this.

The name is ``$ASV_MACHINE``, then ``$NCAR_HOST`` (the cluster, not the node),
then the node name minus trailing digits. Login and compute nodes do not share
a stem, so set ``ASV_MACHINE`` if you use both without ``NCAR_HOST``.

Usage::

    python -m benchmarks.helpers._machine           # record, print the name
    python -m benchmarks.helpers._machine --print   # just print it
"""

import os
import platform
import re
import sys


def default_name():
    for variable in ("ASV_MACHINE", "NCAR_HOST"):
        if os.environ.get(variable):
            return os.environ[variable]
    node = platform.node().split(".")[0]
    return re.sub(r"[-_]?\d+$", "", node) or node


if __name__ == "__main__":
    name = default_name()
    if "--print" not in sys.argv[1:]:
        from asv.machine import Machine, MachineCollection

        MachineCollection.save(name, {**Machine.get_defaults(), "machine": name})
    print(name)
