"""Official four-state primitive proof for every mask in a split manifest."""
import argparse
import json
import os
import shutil
from pathlib import Path
from run_units import ROOT, EF, sha, run

MS=Path('E:/quatra/questa_fse/win64')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--expected-count',type=int,default=110)
    a=p.parse_args();root=a.root.resolve();manifest=a.manifest.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    rows=json.loads(manifest.read_text());assert len(rows)==a.expected_count and a.expected_count>0
    root.mkdir()
    shutil.copy2(__file__,root/'runner.py')
    models=['efx_comb4.v','efx_lut4.v']
    for name in models:shutil.copy2(EF/'sim_models/verilog'/name,root/name)
    # X covers unknown internal values; Z is additionally covered on the two
    # relevant P inputs. P's unused high inputs are constant in mapped LUTs.
    count=len(rows)
    lines=[f'module tb_native_split; reg [3:0] d=0; wire [{count-1}:0] o,p,a,b;']
    for i,row in enumerate(rows):
        m=row['mask'];pm=row['p_mask']
        assert pm==((m>>8)&15)*0x1111
        lines += [f'EFX_COMB4 #(.LUTMASK(16\'h{m:04x}),.MODE("LOGIC")) old{i}(.I0(d[0]),.I1(d[1]),.I2(d[2]),.I3(d[3]),.CI(1\'b0),.O(o[{i}]),.P(p[{i}]),.CO(),.FCO());',
                  f'EFX_LUT4 #(.LUTMASK(16\'h{m:04x})) newO{i}(.I0(d[0]),.I1(d[1]),.I2(d[2]),.I3(d[3]),.O(a[{i}]));',
                  f'EFX_LUT4 #(.LUTMASK(16\'h{pm:04x})) newP{i}(.I0(d[0]),.I1(d[1]),.I2(1\'b0),.I3(1\'b0),.O(b[{i}]));']
    lines += ['''function automatic symbol(input integer v);
 case(v) 0:symbol=0;1:symbol=1;2:symbol=1'bx;default:symbol=1'bz;endcase
 endfunction
 integer checks=0;
 initial begin
  for(integer v=0;v<256;v=v+1)begin
   d={symbol((v>>6)&3),symbol((v>>4)&3),symbol((v>>2)&3),symbol(v&3)};
   #2;
   if(o!==a || p!==b)$fatal(1,"NATIVE_SPLIT_FOURSTATE_MISMATCH input=%b oldO=%h newO=%h oldP=%h newP=%h",d,o,a,p,b);
   checks=checks+220;
  end
  if(checks!=56320)$fatal(1,"NATIVE_SPLIT_COVERAGE");
  $display("PASS NATIVE_SPLIT_110_ALL_FOURSTATE checks=%0d",checks);$finish;
 end
 endmodule''']
    expected_checks=count*256*2
    source='\n'.join(lines).replace('56320',str(expected_checks)).replace('NATIVE_SPLIT_110_ALL_FOURSTATE',f'NATIVE_SPLIT_{count}_ALL_FOURSTATE').replace('checks=checks+220;',f'checks=checks+{count*2};')
    tb=root/'tb_native_split.sv';tb.write_text(source,encoding='utf8')
    meta=dict(manifest=str(manifest),manifest_sha256=sha(manifest),runner_sha256=sha(__file__),
              files={n:sha(root/n) for n in [*models,tb.name]},hardware_access=False)
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    for variable in ('SALT_LICENSE_SERVER','MGLS_LICENSE_FILE','LM_LICENSE_FILE'):
        os.environ[variable]='E:/quatra/LR-195771_License.dat'
    run([MS/'vlib.exe','work'],root,'vlib.log',timeout=15)
    body,_=run([MS/'vlog.exe','-sv',*models,tb.name],root,'compile.log',timeout=20)
    assert 'Errors: 0, Warnings: 0' in body
    body,seconds=run([MS/'vsim.exe','-c','work.tb_native_split','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=30)
    assert f'PASS NATIVE_SPLIT_{count}_ALL_FOURSTATE checks={expected_checks}' in body and 'Errors: 0, Warnings: 0' in body
    result=dict(status=f'LOGIC_SPLIT_{count}_FOURSTATE_{expected_checks}_PASS',seconds=seconds,
                inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),
                scope='Public original COMB4 versus independent O/P Boolean functions. Full fresh-map binding is a separate gate.',
                physical_pass=False,hardware_access=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'])


if __name__=='__main__':main()
