"""Exhaust both corrected byte-adders using the actual official mapped cells."""
import argparse,json,os,pickle,re,shutil,time
from pathlib import Path
from run_units import ROOT,EF,MS,sha,run
from audit_video_service_color_exposure import P
from continuous_control_native import net

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    board=a.board.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    meta=json.loads((board/'inputs.json').read_text());native=Path(meta['native'])
    proof=json.loads((native/'color-exposure-independent-audit.json').read_text())
    audit=json.loads((board/'color-map-audit.json').read_text());assert audit['status']=='COLOR_EXPOSURE_CURRENT_MAP_COMPOSITION_PASS'
    key,g,ps=pickle.loads((board/'color-map-parse.pkl').read_bytes());assert key==sha(board/'outflow/ti60f225_oob.map.v')==audit['map_sha256']
    selected=set(n for c in proof['cones'].values() for n in c['cells']);assert len(selected)==18
    leaves={tuple(t) for c in proof['cones'].values() for t in c['leaves']}
    inputs={'ra':P+'r_bayer_11_00_1P_1','rb':P+'r_bayer_01_00_1P_1','ba':P+'r_bayer_00_01_0P_0','bb':P+'r_bayer_00_00_1P_0'}
    assert leaves=={net(n,i) for n in inputs.values() for i in range(8)}
    ts={t for n in selected for vs in g[n][1].values() for t in vs if t[0]=='net'}
    names={t:'n'+str(i) for i,t in enumerate(sorted(ts,key=str))}
    def token(t):return t[1] if t[0]=='literal' else names[t]
    def param(v):return str(v) if not isinstance(v,tuple) else str(v[0])+"'h"+format(v[1],'x')
    body=['module actual_color(input wire[7:0] ra,rb,ba,bb,output wire[7:0] red,blue);','wire '+','.join(names.values())+';']
    for port,name in inputs.items():
        for bit in range(8):body.append('assign '+names[net(name,bit)]+'='+port+'['+str(bit)+'];')
    for color,port in [('r','red'),('b','blue')]:
        for bit in range(8):body.append('assign '+port+'['+str(bit)+']='+names[net(P+'w_'+color+'_00_00_h0',bit+1)]+';')
    for i,name in enumerate(sorted(selected)):
        kind,pins=g[name];assert kind=='EFX_ADD'
        args=','.join('.'+k+'('+('' if not vs else token(vs[0]) if len(vs)==1 else '{'+','.join(token(t) for t in vs)+'}')+')' for k,vs in pins.items())
        body.append(kind+' #('+','.join('.'+k+'('+param(v)+')' for k,v in ps[name].items())+') c'+str(i)+'('+args+');')
    body.append('endmodule');(root/'actual_color.sv').write_text('\n'.join(body)+'\n')
    tb='''`timescale 1ns/1ps
module tb;
reg[7:0] ra,rb,ba,bb;wire[7:0] red,blue;integer n,m,expected_r,expected_b;
actual_color dut(.*);
initial begin
 for(n=0;n<65536;n=n+1)begin
  ra=n>>8;rb=n; m=(n*40503+71)&65535;ba=m>>8;bb=m;
  expected_r=(int'(ra)+int'(rb))/2;expected_b=(int'(ba)+int'(bb))/2;
  #1;if(red!==8'(expected_r)||blue!==8'(expected_b))$fatal(1,"MAPPED_BAYER n=%0d r=%0d/%0d b=%0d/%0d",n,red,expected_r,blue,expected_b);
 end
 ra='x;rb='x;ba='x;bb='x;#1;
 if(red!==8'hxx||blue!==8'hxx)$fatal(1,"UNKNOWN_PROPAGATION");
 $display("PASS ACTUAL_MAPPED_BAYER_ARITHMETIC comparisons=131072 exhaustive_pairs_per_channel=65536 x_cases=2");$finish;
end
initial begin #70000;$fatal(1,"TIMEOUT");end
endmodule
'''
    (root/'tb.sv').write_text(tb);shutil.copy2(EF/'sim_models/verilog/efx_add.v',root/'efx_add.v')
    inputs_proof={str(f):sha(f) for f in [Path(__file__),board/'color-map-audit.json',native/'color-exposure-independent-audit.json',board/'outflow/ti60f225_oob.map.v',root/'actual_color.sv',root/'tb.sv',root/'efx_add.v']}
    (root/'inputs.json').write_text(json.dumps(dict(evidence=inputs_proof,cells=sorted(selected),leaves=sorted(leaves,key=str)),indent=2))
    os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';start=time.monotonic()
    try:
        run([MS/'vlib.exe','work'],root,'vlib.log',20)
        run([MS/'vlog.exe','-sv','-timescale','1ns/1ps','efx_add.v','actual_color.sv','tb.sv'],root,'compile.log',30)
        body,_=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',30)
        assert 'PASS ACTUAL_MAPPED_BAYER_ARITHMETIC' in body and not re.search(r'\*\* (Error|Fatal)',body)
        for f,h in inputs_proof.items():assert sha(f)==h
        result=dict(status='ACTUAL_MAPPED_BAYER_EXHAUSTIVE_ARITHMETIC_PASS',comparisons=131072,pairs_per_channel=65536,
            independent_operand_sets=True,unknown_cases=2,seconds=time.monotonic()-start,inputs_sha256=sha(root/'inputs.json'),
            scope='Actual mapped arithmetic at registered Bayer sample boundary; does not claim full-frame edge or board picture acceptance',hardware_access=False,board_release=False)
    except Exception as e:
        result=dict(status='FAIL',error=repr(e));raise
    finally:
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
