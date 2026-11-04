"""Exhaustive 0/1/X/Z check of surveyed pairs using official LUT models."""
import argparse,json,os,re,shutil,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from video_service_graph import load
from map_video_service_safe import safe_run

ZERO=('literal',"1'b0",None)
def main():
 p=argparse.ArgumentParser();p.add_argument('--survey',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 out=a.root.resolve();survey=a.survey.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 r=json.loads(survey.read_text());base=Path(r['native']);assert sha(base/'native_source.v')==r['native_sha256']
 D=load();graft=json.loads((base/'graft.json').read_text());edits={}
 for e in graft['changed_loads']:edits.setdefault((e['cell'],e['pin']),{})[tuple(e['from_token'])]=tuple(e['to_token'])
 rows=r['rows'];v=['module tb; reg[3:0] bits; integer x,j; wire['+str(len(rows)-1)+':0] old_value,new_value;']
 for i,row in enumerate(rows):
  boundary=list(map(tuple,row['inputs']));mid=tuple(row['removed_net'])
  def expr(t,allow_mid=False):
   if t[0]=='literal':assert t in (ZERO,('literal',"1'b1",None));return t[1]
   if allow_mid and t==mid:return 'm'+str(i)
   return f'bits[{boundary.index(t)}]' if t in boundary else "1'b0"
  def pins(name,allow_mid=False):
   ps=D['g'][name][1]
   return ','.join('.I'+str(n)+'('+expr(edits.get((name,'I'+str(n)),{}).get(ps.get('I'+str(n),[ZERO])[0],ps.get('I'+str(n),[ZERO])[0]),allow_mid)+')' for n in range(4))
  v += [f'wire m{i};',f'EFX_LUT4 #(.LUTMASK(16\'h{row["original_parent_mask"]:04x})) p{i}({pins(row["parent"])},.O(m{i}));',
        f'EFX_LUT4 #(.LUTMASK(16\'h{row["original_child_mask"]:04x})) c{i}({pins(row["child"],True)},.O(old_value[{i}]));',
        f'EFX_LUT4 #(.LUTMASK(16\'h{row["lutmask"]:04x})) n{i}('+','.join(f'.I{n}('+ (f'bits[{n}]' if n<len(boundary) else "1'b0")+')' for n in range(4))+f',.O(new_value[{i}]));']
 v+=['initial begin for(x=0;x<256;x=x+1)begin for(j=0;j<4;j=j+1)begin case((x>>(j*2))&3) 0:bits[j]=0;1:bits[j]=1;2:bits[j]=1\'bx;3:bits[j]=1\'bz;endcase end #2;']
 for i,row in enumerate(rows):v.append(f'if(old_value[{i}]!==new_value[{i}])$display("MISMATCH pair={i} stimulus=%d old=%b new=%b",x,old_value[{i}],new_value[{i}]);')
 v += ['$display("VECTOR_DONE %0d",x);end $display("EXHAUSTIVE_DONE");$finish;end endmodule']
 out.mkdir();(out/'tb.sv').write_text('\n'.join(v));shutil.copy2(EF/'sim_models/verilog/efx_lut4.v',out/'efx_lut4.v')
 ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';start=time.monotonic()
 safe_run([ms/'vlib.exe','work'],out,'vlib.log',20)
 safe_run([ms/'vlog.exe','-sv','-timescale','1ns/1ps','efx_lut4.v','tb.sv'],out,'compile.log',45)
 body,_=safe_run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1};onbreak {quit -code 1};run -all;quit -code 0'],out,'simulate.log',60)
 assert 'EXHAUSTIVE_DONE' in body and len(re.findall(r'^# VECTOR_DONE ',body,re.M))==256
 bad={int(x) for x in re.findall(r'MISMATCH pair=(\d+)',body)}
 result=dict(status='OFFICIAL_LUT_FOUR_STATE_FILTER_COMPLETED',pairs=len(rows),vectors_per_pair=256,accepted=[i for i in range(len(rows)) if i not in bad],rejected=sorted(bad),
  survey=str(survey),survey_sha256=sha(survey),source_sha256=r['native_sha256'],seconds=time.monotonic()-start,
  inputs={str(f):sha(f) for f in [Path(__file__),survey,base/'native_source.v',out/'tb.sv',out/'efx_lut4.v']},hardware_access=False)
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','accepted')},indent=2))
if __name__=='__main__':main()
