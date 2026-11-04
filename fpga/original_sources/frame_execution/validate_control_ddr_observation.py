"""Actual mapped OFP3 status/event path with functional control held in reset.

The DDR source status/enable boundary is stimulated, not a DDR PHY model.
"""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,sha
from read_control_ddr_progress import decode
from windows_process_tree import stop_launched_tree
Q=Path('E:/quatra/questa_fse/win64')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    src=a.source.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    ex=json.loads((src/'extraction.json').read_text());assert sha(src/'mapped_control.v')==ex['cone_sha256']
    root.mkdir()
    for f in src.glob('*.v'):shutil.copy2(f,root/f.name)
    ports=ex['inputs'];lines=['`timescale 1ns/1ps','module tb;'];connections=[]
    for name,port in ports.items():
        lines.append(f'reg {ex["input_widths"][name]} {port}=0;');connections.append(f'.{port}({port})')
    for name in ex['outputs']:lines.append(f'wire {name};');connections.append(f'.{name}({name})')
    lines.append('mapped_control dut('+','.join(connections)+');')
    body=r'''
always #3.333 CLOCK=~CLOCK;
task tick;begin #50;TCK=1;#50;TCK=0;end endtask
task scan(input [449:0] packet,output [449:0] raw);
 begin SEL=1;SHIFT=0;CAP=1;tick;CAP=0;SHIFT=1;raw=0;
 for(integer i=0;i<450;i=i+1)begin TDI=packet[i];#1;raw[i]=jtag_inst1_TDO;tick;end
 SHIFT=0;tick;SEL=0;tick;end
endtask
reg [449:0] raw;
task snap;
 begin scan(2,raw);scan(3,raw);
 if(raw[33:2]!==32'h4f465033 || raw[35:34]!==1 || raw[49:42]!==3)$fatal(1,"snapshot handshake/header");
 $display("DDRRAW %h",raw);
 if(dut.\cpu/observation_accepts !==0 || dut.\cpu/observation_replies !==0)$fatal(1,"observation issued commands");
 end
endtask
initial begin
 // Keep the real external reset asserted: this path must work despite APB reset.
 STATUS=32'h001bff5f;#1000;
 REQUEST=36'h2000;#100;REQUEST=0;#100;
 FORWARD=1;#100;FORWARD=0;#100;
 snap;
 STATUS=32'h001b00f6;QUALIFIED=1;#1000;snap;
 $display("PASS MAPPED_DDR_OBSERVATION_INDEPENDENT_OF_CONTROL");$finish;
end
initial begin #300000;$fatal(1,"DDR observation simulation limit");end
endmodule
'''
    replace={'CLOCK':ports['process_clk_150'],'TCK':ports['jtag_inst1_TCK'],'TDI':ports['jtag_inst1_TDI'],
        'SEL':ports['jtag_inst1_SEL'],'CAP':ports['jtag_inst1_CAPTURE'],'SHIFT':ports['jtag_inst1_SHIFT'],
        'STATUS':ports['bringup_ddr_sync'],'REQUEST':ports['independent_ddr_debug'],
        'FORWARD':ports['pll_shift_ena'],'QUALIFIED':ports['cal_done']}
    body=re.sub(r'\b('+'|'.join(replace)+r')\b',lambda m:replace[m[0]],body)
    (root/'tb.sv').write_text('\n'.join(lines)+body)
    meta=dict(source=str(src),source_map_sha256=ex['source_sha256'],source_extraction_sha256=sha(src/'extraction.json'),
        files={f.name:sha(f) for f in root.iterdir() if f.is_file()},tck_hz=10000000,hardware_access=False)
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';start=time.monotonic()
    for args,name,budget in [(['vlib.exe','work'],'vlib.log',15),
        (['vlog.exe','-sv','-timescale','1ns/1ps',*[f.name for f in root.glob('*.v')],'tb.sv'],'compile.log',35),
        (['vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],'simulation.log',90)]:
        with (root/name).open('x') as f:
            proc=subprocess.Popen([str(Q/args[0]),*args[1:]],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
            try:code=proc.wait(timeout=budget)
            except subprocess.TimeoutExpired:stop_launched_tree(proc);raise
        log=(root/name).read_text(errors='replace');assert code==0 and not re.search(r'(?m)^#?\s*\*\* (Error|Fatal)',log),log[-2000:]
    assert 'PASS MAPPED_DDR_OBSERVATION_INDEPENDENT_OF_CONTROL' in log
    records=[decode(int(v,16)) for v in re.findall(r'DDRRAW ([\da-fA-F]+)',log)]
    assert len(records)==2
    fail,success=(r['ddr'] for r in records)
    assert fail['raw_done'] and not fail['raw_pass'] and fail['failure_log']==255
    assert not records[0]['levels']['ddr_qualified'] and records[1]['levels']['ddr_qualified']
    assert success['raw_done'] and success['raw_pass'] and success['failure_log']==0
    assert all(r['levels']['control_reset'] for r in records)
    assert fail['sampled_shift_requests']==fail['sampled_shift_forwarded']==1
    assert success['sampled_system_reset_events']==success['sampled_ddr_lock_drops']==1
    for name,digest in meta['files'].items():assert sha(root/name)==digest
    (root/'result.json').write_text(json.dumps(dict(status='MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS',
        seconds=time.monotonic()-start,records=records,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),
        scope='Actual mapped observer and serial path; stimulated DDR public status boundary, not PHY calibration',hardware_access=False),indent=2))
    print('MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS')
if __name__=='__main__':main()
