"""Use the same tested recovery engine in a separate calibrated-pruning run."""
from . import recover_pruned_teacher as recovery
from .calibrate_teacher_pruning import RUN

if __name__=='__main__':
    recovery.RUN=RUN
    recovery.main()
