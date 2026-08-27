"""SMP downstream RL tasks.

Importing this package registers all SMP tasks in ``mjlab.tasks.registry``
via side-effect imports of each task sub-package.
"""

from smp.rl.tasks import (
  ame,  # noqa: F401  # registers AME-G1 tasks
  cmoe,  # noqa: F401  # registers CMoE-G1
  getup,  # noqa: F401  # registers Smp-Getup-G1
  location,  # noqa: F401  # registers Smp-Location-G1
  steering,  # noqa: F401  # registers Smp-Steering-G1 and Smp-Forward-G1
)
