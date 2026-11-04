"""Publish calibrated pruning without modifying historical run artifacts."""
from . import publish_pruned_teacher as publisher
from .calibrate_teacher_pruning import RUN

if __name__=='__main__':
    publisher.RUN=RUN
    publisher.main()
