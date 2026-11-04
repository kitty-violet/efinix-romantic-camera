"""Check both proposed physical-cell outputs against official four-state LUTs."""
import argparse, json, os, re, shutil, time
from pathlib import Path
from run_units import ROOT, EF, sha
from map_video_service_safe import safe_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--survey',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root.resolve();survey=a.survey.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
    r=json.loads(survey.read_text());base=Path(r['native']);assert sha(base/'native_source.v')==r['native_sha256']
    rows=r['rows'];assert rows
    v=[f'module tb; reg[3:0] bits; integer x,j; wire[{len(rows)-1}:0] old_o,old_p,new_o,new_p;']
    for i,row in enumerate(rows):
        boundary=list(map(tuple,row['inputs']))
        def expr(t):
            t=tuple(t)
            if t[0]=='literal':assert t[1] in ("1'b0","1'b1");return t[1]
            return f'bits[{boundary.index(t)}]'
        for role,output in [('primary','o'),('secondary','p')]:
            pins=','.join(f'.I{j}({expr(t)})' for j,t in enumerate(row[role+'_inputs']))
            v.append(f'EFX_LUT4 #(.LUTMASK(16\'h{row[role+"_mask"]:04x})) old_{role}_{i}({pins},.O(old_{output}[{i}]));')
        pins=','.join(f'.I{j}('+ (f'bits[{j}]' if j<len(boundary) else "1'b0")+')' for j in range(4))
        v.append(f'EFX_COMB4 #(.LUTMASK(16\'h{row["lutmask"]:04x}),.MODE("LOGIC")) pair_{i}({pins},.CI(1\'b0),.O(new_o[{i}]),.P(new_p[{i}]),.CO(),.FCO());')
    v+=['initial begin for(x=0;x<256;x=x+1)begin for(j=0;j<4;j=j+1)begin case((x>>(j*2))&3) 0:bits[j]=0;1:bits[j]=1;2:bits[j]=1\'bx;3:bits[j]=1\'bz;endcase end #2;']
    for i in range(len(rows)):
        v.append(f'if({{old_o[{i}],old_p[{i}]}}!=={{new_o[{i}],new_p[{i}]}})$display("MISMATCH pair={i} stimulus=%d old=%b%b new=%b%b",x,old_o[{i}],old_p[{i}],new_o[{i}],new_p[{i}]);')
    v+=['$display("VECTOR_DONE %0d",x);end $display("EXHAUSTIVE_DONE");$finish;end endmodule']
    out.mkdir();(out/'tb.sv').write_text('\n'.join(v),encoding='utf8')
    for name in ('efx_lut4.v','efx_comb4.v'):shutil.copy2(EF/'sim_models/verilog'/name,out/name)
    ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';start=time.monotonic()
    safe_run([ms/'vlib.exe','work'],out,'vlib.log',20)
    safe_run([ms/'vlog.exe','-sv','-timescale','1ns/1ps','efx_lut4.v','efx_comb4.v','tb.sv'],out,'compile.log',45)
    body,_=safe_run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1};onbreak {quit -code 1};run -all;quit -code 0'],out,'simulate.log',60)
    assert 'EXHAUSTIVE_DONE' in body and len(re.findall(r'^# VECTOR_DONE ',body,re.M))==256
    bad={int(x) for x in re.findall(r'MISMATCH pair=(\d+)',body)}
    result=dict(status='OFFICIAL_LOGIC_PAIR_FOUR_STATE_FILTER_COMPLETED',pairs=len(rows),vectors_per_pair=256,outputs_per_pair=2,
        accepted=[i for i in range(len(rows)) if i not in bad],rejected=sorted(bad),survey=str(survey),survey_sha256=sha(survey),
        source_sha256=r['native_sha256'],seconds=time.monotonic()-start,
        inputs={str(f):sha(f) for f in [Path(__file__),survey,base/'native_source.v',out/'tb.sv',out/'efx_lut4.v',out/'efx_comb4.v']},hardware_access=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','accepted')},indent=2))


if __name__=='__main__':main()
