"""Resolve a recorded source key without changing its recorded digest."""
import os
from pathlib import Path
from run_units import ROOT, sha


def recorded_source_digest(evidence, target):
    target = Path(target).resolve()
    assert target.is_relative_to(ROOT), target
    def identity(value):
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        return os.path.normcase(str(path.resolve()))
    matches = [(key, value) for key, value in evidence.items()
               if identity(key) == identity(target)]
    assert len(matches) == 1, ('Missing or ambiguous recorded source', str(target), matches)
    key, digest = matches[0]
    assert sha(target) == digest, ('Recorded source changed', key)
    return digest
