"""Official four-state models for actual freshly mapped replicated drivers."""
import argparse,json,os,shutil
from pathlib import Path
from run_units import ROOT,EF,sha,run
from probe_full_native_split_map import params
from audit_snapshot_cells import cells
MS=Path('E:/quatra/questa_fse/win64')

def main():
    p=argparse.ArgumentParser();p.add_argument('--map',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();mapped=a.map.resolve();root=a.root.resolve()
    r=json.loads((mapped/'result.json').read_text());assert r['status']=='FULL_NATIVE_LOCAL_DRIVERS_MAP_GRAPH_PASS'
    net=mapped/'outflow/ti60f225_oob.map.v';assert sha(net)==r['map_sha256']
    rows=json.loads((mapped/'replications.json').read_text());text=net.read_text();ps=params(text);cc=cells(text)
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    for n in ('efx_lut4.v','efx_ff.v'):shutil.copy2(EF/'sim_models/verilog'/n,root/n)
    count=sum(len(row['clones']) for row in rows)
    lines=[f'module tb_local_drivers; reg [3:0] d=0; reg clk=0; wire [{len(rows)-1}:0] old; wire [{count-1}:0] q; integer checks=0;']
    index=0
    for group,row in enumerate(rows):
        original=row['driver'];kind=row['kind']
        for n,output in [(original,f'old[{group}]')]+[(x['name'],f'q[{index+i}]') for i,x in enumerate(row['clones'])]:
            assert cc[n][0]==kind
            definitions=','.join(f".{k}({w}'h{v:x})" for k,(w,v) in ps[n].items())
            if kind=='EFX_LUT4':ports=','.join(f'.I{i}(d[{i}])' for i in range(4))+f',.O({output})'
            else:ports=f'.D(d[0]),.CE(d[1]),.SR(d[2]),.CLK(clk),.Q({output})'
            lines.append(f'{kind} #({definitions}) cell_{group}_{output.replace("[","_").replace("]","")}({ports});')
        index+=len(row['clones'])
    lines.append('''function automatic symbol(input integer v);
      case(v) 0:symbol=0;1:symbol=1;2:symbol=1'bx;default:symbol=1'bz;endcase
      endfunction
      initial begin
      for(integer v=0;v<256;v=v+1) begin
       clk=0; d={symbol((v>>6)&3),symbol((v>>4)&3),symbol((v>>2)&3),symbol(v&3)};
       #2; clk=1; #2;
    ''')
    index=0
    for group,row in enumerate(rows):
        for x in row['clones']:
            lines.append(f'if(q[{index}]!==old[{group}]) $fatal(1,"LOCAL_DRIVER_FOURSTATE {x["name"]} v=%d",v); checks=checks+1;');index+=1
    lines += ['end',f'if(checks!={count*256})$fatal(1,"CHECK_COUNT");',f'$display("PASS LOCAL_DRIVERS_FOURSTATE checks=%0d",checks); $finish; end endmodule']
    tb=root/'tb_local_drivers.sv';tb.write_text('\n'.join(lines))
    meta=dict(map_sha256=sha(net),map_result_sha256=sha(mapped/'result.json'),replications_sha256=sha(mapped/'replications.json'),
        runner_sha256=sha(__file__),files={n:sha(root/n) for n in ('efx_lut4.v','efx_ff.v',tb.name)},hardware_access=False)
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    for v in ('SALT_LICENSE_SERVER','MGLS_LICENSE_FILE','LM_LICENSE_FILE'):os.environ[v]='E:/quatra/LR-195771_License.dat'
    run([MS/'vlib.exe','work'],root,'vlib.log',timeout=15)
    body,_=run([MS/'vlog.exe','-sv','efx_lut4.v','efx_ff.v',tb.name],root,'compile.log',timeout=25)
    assert 'Errors: 0, Warnings: 0' in body
    body,seconds=run([MS/'vsim.exe','-c','work.tb_local_drivers','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=40)
    assert f'PASS LOCAL_DRIVERS_FOURSTATE checks={count*256}' in body and 'Errors: 0, Warnings: 0' in body
    result=dict(status='ACTUAL_MAP_LOCAL_DRIVERS_FOURSTATE_PASS',checks=count*256,seconds=seconds,
        inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),hardware_access=False,
        scope='Identical mapped LUT/FF primitive parameters under common four-state inputs. Reversible connectivity and full board shorts are separate gates.',board_release=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
