"""Bind Windows or POSIX artifact receipts without accepting path aliases."""
from pathlib import Path, PureWindowsPath


def checked_artifact(artifacts, root, target):
    root = Path(root).resolve()
    target = Path(target).resolve()
    assert target.is_relative_to(root), 'Artifact target escaped its root'
    normalized = {}
    for key, record in artifacts.items():
        path = PureWindowsPath(key)
        assert not path.drive and not path.root, ('Absolute artifact path', key)
        assert path.parts and '..' not in path.parts, ('Invalid artifact path', key)
        resolved = root.joinpath(*path.parts).resolve()
        assert resolved.is_relative_to(root), ('Artifact escaped its root', key)
        assert resolved not in normalized, ('Duplicate artifact path', key)
        normalized[resolved] = record
    assert target in normalized, ('Missing artifact receipt', str(target))
    return normalized[target]
