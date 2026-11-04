"""Remove only proved unobservable native primitive cones in a new copy."""
import argparse,collections,json,re,shutil,subprocess,sys
from pathlib import Path
from native_local_drivers import CELL
from run_units import ROOT,HERE,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--physical',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();base=a.base.resolve();out=a.root.resolve();physical=a.physical.resolve()
    assert out.is_relative_to(ROOT/'tmp/frame-execution') and not out.exists()
    proof_path=out.parent/(out.name+'-unused-audit.json');assert not proof_path.exists()
    subprocess.run([sys.executable,str(HERE/'survey_video_service_native_unused.py'),
                    '--native',str(base),'--physical',str(physical),'--output',str(proof_path)],check=True,timeout=240)
    proof=json.loads(proof_path.read_text());src=base/'native_source.v'
    assert proof['status']=='NATIVE_PUBLIC_UNUSED_CONES_CANDIDATES_ONLY' and proof['source_sha256']==sha(src)
    assert proof['constraints_sha256']==sha(physical/'ti60f225_oob.sdc')
    assert proof['anchors_sha256']==sha(physical/'fixed_dsp.placeloc')
    dead=set(proof['cells']);assert dead and len(dead)==proof['candidate_count']
    text=src.read_text();seen=set();kinds=collections.Counter()
    def drop(m):
        name=m[2].lstrip('\\')
        if name not in dead:return m[0]
        assert name not in seen and m[1] in ('EFX_LUT4','EFX_FF','EFX_ADD','EFX_SRL8')
        seen.add(name);kinds[m[1]]+=1;return ''
    text=CELL.sub(drop,text);assert seen==dead and dict(kinds)==proof['kinds']
    text=re.sub(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',
                lambda m:'' if m[1].lstrip('\\') in dead else m[0],text,flags=re.M)
    graft=json.loads((base/'graft.json').read_text())
    assert not dead.intersection(graft['removed'])
    graft['removed']=sorted(set(graft['removed'])|dead)
    graft['removed_kinds']=dict(collections.Counter(graft['removed_kinds'])+kinds)
    graft['changed_loads']=[r for r in graft['changed_loads'] if r['cell'] not in dead]
    graft['retained_primitive_count']-=len(dead)
    graft['unused_pruning']=dict(status='PUBLIC_OBSERVABILITY_CUT_APPLIED_REQUIRES_MAP_AND_PHYSICAL_CHECKS',
                                base=str(base),base_sha256=sha(src),proof=str(proof_path),proof_sha256=sha(proof_path),
                                removed=sorted(dead),actual_area_savings_unmeasured=True)
    for f in (Path(__file__),HERE/'survey_video_service_native_unused.py',base/'graft.json',proof_path):
        graft['inputs'][str(f)]=sha(f)
    out.mkdir()
    for f in base.glob('*.sv'):shutil.copy2(f,out/f.name)
    (out/'native_source.v').write_text(text,encoding='utf8')
    (out/'graft.json').write_text(json.dumps(graft,indent=2),encoding='utf8')
    print(json.dumps(dict(status=graft['unused_pruning']['status'],removed=len(dead),kinds=dict(kinds),
                         retained=graft['retained_primitive_count'],hardware_access=False)))

if __name__=='__main__':main()
