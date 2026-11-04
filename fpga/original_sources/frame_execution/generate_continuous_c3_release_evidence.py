"""Materialize the complete existing release checks with three exact C3 bindings."""
import builtins, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'finish_continuous_control_release2.py'
OUTPUT = HERE/'continuous_c3_release_evidence.py'
META = HERE/'continuous_c3_release_evidence.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def generate():
    captured = []
    def capture(source, filename, mode, *args, **kwargs):
        if isinstance(source, str) and str(filename).endswith(':joint-closure-release'):
            captured.append(source)
        return builtins.compile(source, filename, mode, *args, **kwargs)
    scope = dict(__name__='capture_release_without_execution', __file__=str(SOURCE), compile=capture)
    exec(builtins.compile(SOURCE.read_text(encoding='utf8'), str(SOURCE), 'exec'), scope)
    assert len(captured) == 1
    original = captured[0]
    code = original
    changes = []
    for old, new, count in (
        ("dict(before='CONGESTION_2',after='CONGESTION_2')", "dict(before='CONGESTION_3',after='CONGESTION_3')", 1),
        ('continuous_control_place.py', 'continuous_control_c3_place.py', 3),
        ('continuous_control_route', 'continuous_control_c3_route', 5)):
        actual = code.count(old)
        assert actual == count, (old, actual, count)
        code = code.replace(old, new)
        changes.append(dict(old=old,new=new,occurrences=count))
    builtins.compile(code, str(OUTPUT), 'exec')
    return code, dict(status='COMPLETE_CURRENT_RELEASE_GATES_REBOUND_TO_C3',source=str(SOURCE),source_sha256=sha(SOURCE),generator_sha256=sha(__file__),original_generated_sha256=hashlib.sha256(original.encode()).hexdigest(),generated_sha256=hashlib.sha256(code.encode()).hexdigest(),changes=changes,checks_removed=False,hardware_access=False)

def verify_generated():
    code, meta = generate()
    assert OUTPUT.read_text(encoding='utf8') == code
    assert json.loads(META.read_text()) == meta
    return meta

if __name__ == '__main__':
    if sys.argv[1:] == ['--check-only']:
        verify_generated(); print('COMPLETE_C3_RELEASE_GATES_REGENERATION_PASS')
    else:
        assert not sys.argv[1:] and not OUTPUT.exists() and not META.exists()
        code, meta = generate()
        OUTPUT.write_text(code, encoding='utf8')
        META.write_text(json.dumps(meta,indent=2), encoding='utf8')
        print('COMPLETE_C3_RELEASE_GATES_GENERATED')
