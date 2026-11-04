"""Compare official encrypted CPU operators with Efinity-exported operators.

Produces a simulation-only hybrid CPU: untouched public vendor RTL plus the
two official mapped operators. No guessed operator, decryption or production
source change. The directed/random comparison is not a formal equivalence proof.
"""
import argparse
import json
import re
import shutil
import traceback
from pathlib import Path
from run_units import ROOT, EF, MS, sha, run

TB = r'''`timescale 1ns/1ps
module tb_cpu_helpers;
reg[31:0] a=0,b=0,addsub=0;reg[1:0] bitctrl=0,ctrl=0,shiftctrl=0;reg less=0;
wire[31:0] old1,new1,old2,new2;
EfxCPUSp1 original1(.src1(a),.src2(b),.addSub(addsub),.bitCtrl(bitctrl),.ctrl(ctrl),.less(less),.result(old1));
mapped_EfxCPUSp1 mapped1(.src1(a),.src2(b),.addSub(addsub),.bitCtrl(bitctrl),.ctrl(ctrl),.less(less),.result(new1));
EfxCPUSp2 original2(.src1(a),.src2(b),.ctrl(shiftctrl),.result(old2));
mapped_EfxCPUSp2 mapped2(.src1(a),.src2(b),.ctrl(shiftctrl),.result(new2));
integer seed=13059,n;initial begin
 for(n=0;n<131072;n=n+1)begin
  ctrl=n[1:0];bitctrl=n[3:2];less=n[4];shiftctrl=n[6:5];
  if(n<1024)begin
   case(n[9:7])
    0:begin a=0;b=0;end
    1:begin a=32'hffffffff;b=1;end
    2:begin a=32'h80000000;b=31;end
    3:begin a=32'h7fffffff;b=32;end
    4:begin a=32'h55555555;b=32'haaaaaaaa;end
    5:begin a=1;b=32'hffffffff;end
    6:begin a=32'hffffffff;b=32'hffffffff;end
    7:begin a=32'h80000000;b=32'h80000000;end
   endcase
  end else begin a=$random(seed);b=$random(seed);end
  addsub=n[7] ? a-b : a+b;
  #1;
  if(old1!==new1||old2!==new2)$fatal(1,"operator mismatch n=%0d original=%h/%h mapped=%h/%h",n,old1,old2,new1,new2);
 end
 $display("PASS CPU_HELPERS_COMPARE samples=131072 operators=2 all_control_combinations=1");$finish;
end
endmodule
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--maps',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path)
    a=p.parse_args();root=a.root.resolve();maps=a.maps.resolve();ref=a.reference.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir(parents=True)
    try:
        records=[]
        def copy(f):
            dst=root/f.name;assert not dst.exists();shutil.copy2(f,dst)
            records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)));return dst
        original=copy(ref/'EfxSapphireSoc.v')
        public=original.read_text().split('module EfxCPUSp1 (',1)[0]
        assert '`protected' not in public and len(public)>100000
        kinds=set();networks=[];map_records=[]
        for name in ('EfxCPUSp1','EfxCPUSp2'):
            meta=json.loads((maps/name/'result.json').read_text());net=maps/name/'outflow/board_ip.map.v'
            assert meta['map_complete'] and meta['net_sha256']==sha(net)
            body=net.read_text()
            assert not re.search(r'`protect|pragma\s+protect',body), 'Official export retains protected_cell; native helper acceleration unavailable'
            assert re.search(r'\bmodule\s+'+name+r'\b',body)
            kinds.update(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',body,re.M))
            networks.append(body)
            alias=re.sub(r'\bmodule\s+'+name+r'\b','module mapped_'+name,body,count=1)
            dst=root/(name+'.mapped.v');dst.write_text(alias)
            map_records.append(dict(source=str(net),sha256=sha(net),renamed_sha256=sha(dst)))
        primitives=[copy(EF/'sim_models/verilog'/f'efx_{k.lower()}.v') for k in sorted(kinds)]
        (root/'tb_cpu_helpers.sv').write_text(TB)
        (root/'cpu_hybrid.v').write_text(public+'\n'+'\n'.join(networks))
        inputs=dict(inputs=records,maps=map_records,tb_sha256=sha(root/'tb_cpu_helpers.sv'),
                    hybrid_sha256=sha(root/'cpu_hybrid.v'),runner_sha256=sha(__file__),primitive_kinds=sorted(kinds))
        (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
        run([MS/'vlib.exe','work'],root,'vlib.log')
        run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',original,*primitives,
             root/'EfxCPUSp1.mapped.v',root/'EfxCPUSp2.mapped.v',root/'tb_cpu_helpers.sv'],root,'compile.log')
        body,seconds=run([MS/'vsim.exe','-c','work.tb_cpu_helpers','-do',
                          'onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
        assert 'PASS CPU_HELPERS_COMPARE' in body
        for r in records:assert sha(r['snapshot'])==r['sha256']
        result=dict(status='CPU_HELPERS_DIRECTED_RANDOM_COMPARE_PASS',samples=131072,operators=2,seconds=seconds,
                    inputs_sha256=sha(root/'inputs.json'),hybrid_sha256=sha(root/'cpu_hybrid.v'),
                    log_sha256=sha(root/'simulation.log'),formal_equivalence=False,full_cpu_validated=False,
                    production_rtl_modified=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()
