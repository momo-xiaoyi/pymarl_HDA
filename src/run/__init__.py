from .run import run
from .run2 import run as run2
from .run_offpg import run as run_offpg
REGISTRY = {}

REGISTRY["run"] = run
REGISTRY["run2"] = run2
REGISTRY["run_offpg"] = run_offpg

from .run_pp import run as run_pp
REGISTRY["run_pp"] = run_pp
