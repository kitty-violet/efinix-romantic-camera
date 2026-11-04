"""Inline every PUBLIC LUT model verbatim in this simulation-only cone.

This changes hierarchy only: the vendor function, weak defaults, initialization,
sensitivity list and timescale remain intact. No protected model is decoded,
replaced or bypassed; no synthesized/routed file is changed.
"""
import inspect,re
from run_units import sha
import inline_official_state_luts as old
def inline(text,model,predicate):
    names=re.findall(r'\bEFX_LUT4\s+(\S+)\s*\(',text)
    assert len([n for n in names if predicate(n)])==23
    assert len(names)>23 and len(set(names))==len(names)
    code=inspect.getsource(old.inline)
    assert code.count('assert len(selected)==23,len(selected)')==1
    code=code.replace('assert len(selected)==23,len(selected)',f'assert len(selected)=={len(names)},len(selected)')
    code=code.replace('primitive_instances_removed=23','primitive_instances_removed=len(selected)')
    scope=dict(old.__dict__);exec(compile(code,old.__file__+':all-public-LUTs','exec'),scope)
    changed,proof=scope['inline'](text,model,lambda n:True)
    assert not re.search(r'\bEFX_LUT4\s+\S+\s*\(',changed)
    proof.update(transform_sha256=sha(__file__),parent_transform_sha256=sha(old.__file__),
        scope='Every retained public LUT in the test cone, copied exact official body; original map/clock/stimulus/assertions unchanged')
    return changed,proof
