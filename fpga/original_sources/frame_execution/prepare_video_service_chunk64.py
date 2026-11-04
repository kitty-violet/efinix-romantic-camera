"""Independent native+test copies; change only the serializer implementation."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);a=p.parse_args()
 base=ROOT/'tmp/frame-execution/video-service-native13';ref=ROOT/'tmp/frame-execution/video-service-serial-boundary-native1'
 source=HERE/'variants/video_service_chunk64/video_snapshot_serializer.sv'
 target=a.native.resolve();fixture=a.fixture.resolve()
 for out in (target,fixture):assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 audit=json.loads((base/'native-graph-audit.json').read_text())
 for f,h in audit['inputs'].items():assert sha(f)==h,f
 old=json.loads((ref/'inputs.json').read_text())
 for f,h in old['files'].items():assert sha(f)==h,f
 target.mkdir();fixture.mkdir()
 copied={}
 for f in base.iterdir():
  if f.is_file() and f.name not in ('native-graph-audit.json','graft.json'):
   shutil.copy2(f,target/f.name);copied[f.name]=sha(f)
 shutil.copy2(source,target/source.name)
 graft=json.loads((base/'graft.json').read_text())
 graft['inputs'].update({str(source):sha(source),str(Path(__file__)):sha(__file__)})
 graft['serializer_implementation']=dict(kind='64bit_reload_shift',source=str(source),sha256=sha(source),parent_graft_sha256=sha(base/'graft.json'))
 (target/'graft.json').write_text(json.dumps(graft,indent=2))
 assert all(sha(target/n)==h for n,h in copied.items() if n!=source.name)
 for f in ref.iterdir():
  if f.suffix in ('.v','.sv'):shutil.copy2(f,fixture/f.name)
 shutil.copy2(source,fixture/source.name)
 files={str(f):sha(f) for f in fixture.iterdir()};files.update({str(source):sha(source),str(Path(__file__)):sha(__file__)})
 (fixture/'inputs.json').write_text(json.dumps(dict(files=files,parent_inputs_sha256=sha(ref/'inputs.json'),only_dut_change='video_snapshot_serializer; same boundary stimulus',hardware_access=False),indent=2))
 (target/'candidate_derivation.json').write_text(json.dumps(dict(parent=str(base),parent_graft_sha256=sha(base/'graft.json'),parent_files=copied,changed_source=str(source),changed_source_sha256=sha(source),generator_sha256=sha(__file__),hardware_access=False),indent=2))
 print('CHUNK64_NATIVE_AND_SHORT_PREPARED')
if __name__=='__main__':main()
