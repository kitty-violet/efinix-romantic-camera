"""Retain every complete physical gate, rebinding only current artifacts."""
import builtins,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'finish_continuous_control_release2.py'
OUTPUT=HERE/'continuous_srl_release_evidence.py'
META=HERE/'continuous_srl_release_evidence.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def generate():
    captured=[]
    def capture(source,filename,mode,*args,**kwargs):
        if isinstance(source,str) and str(filename).endswith(':joint-closure-release'):captured.append(source)
        return builtins.compile(source,filename,mode,*args,**kwargs)
    scope=dict(__name__='capture_only',__file__=str(SOURCE),compile=capture)
    exec(builtins.compile(SOURCE.read_text(encoding='utf8'),str(SOURCE),'exec'),scope)
    assert len(captured)==1
    original=captured[0];code=original;changes=[]
    for old,new,count in (
        ('from continuous_control_board import BOARD,audit_delta,TRANSFORM','from continuous_srl_register_board import BOARD,audit_delta,TRANSFORM',1),
        ('continuous_control_place.py','continuous_srl_register_place.py',3),
        ('continuous_control_route','continuous_srl_register_route',5),
        ('continuous-control-evidence1','continuous-srl-register-evidence1',2)):
        assert code.count(old)==count,(old,code.count(old),count)
        code=code.replace(old,new);changes.append(dict(old=old,new=new,occurrences=count))
    # The former generator executed this body in its outer namespace. A saved
    # module must explicitly import the two recovery-gate dependencies.
    needle='from run_units import ROOT,HERE,sha\n'
    assert code.count(needle)==1
    code=code.replace(needle,needle+'from audit_legacy_cache_recovery import MANIFEST,prove\n')
    changes.append(dict(kind='explicit_outer_namespace_import',names=['MANIFEST','prove'],checks_removed=False))
    builtins.compile(code,str(OUTPUT),'exec')
    return code,dict(status='COMPLETE_PHYSICAL_GATES_REBOUND_TO_SRL_REGISTER_MAP',source=str(SOURCE),source_sha256=sha(SOURCE),generator_sha256=sha(__file__),original_generated_sha256=hashlib.sha256(original.encode()).hexdigest(),generated_sha256=hashlib.sha256(code.encode()).hexdigest(),changes=changes,checks_removed=False,functional_parent_identity_preserved=True,hardware_access=False)
def verify_generated():
    code,meta=generate()
    assert OUTPUT.read_text(encoding='utf8')==code and json.loads(META.read_text())==meta
    return meta
if __name__=='__main__':
    if sys.argv[1:]==['--check-only']:verify_generated();print('SRL_FULL_GATES_REGENERATION_PASS')
    else:
        assert not sys.argv[1:] and not OUTPUT.exists() and not META.exists()
        code,meta=generate();OUTPUT.write_text(code,encoding='utf8');META.write_text(json.dumps(meta,indent=2),encoding='utf8');print('SRL_COMPLETE_GATES_GENERATED')
