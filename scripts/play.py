"""Play wrapper for SMP tasks."""

from __future__ import annotations

import sys

import smp.rl.tasks  # noqa: F401  # registers Smp-* tasks in the mjlab registry

if __name__ == "__main__":
  if any(argument.startswith("--attention-file") for argument in sys.argv):
    from smp.rl.ame.play import main
  else:
    from mjlab.scripts.play import main
  main()
