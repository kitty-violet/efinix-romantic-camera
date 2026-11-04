"""Bounded full-geometry ingress test for the observed 803-row short frame.

Checks RAW pack/FIFO/unpack/line feeder and cancellation handshake only.
Does not claim camera configuration, ISP, DDR or board stall reproduction.
"""
import argparse, json, os, re, shutil, time
from pathlib import Path
from run_units import ROOT, HERE, run, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    base=ROOT/'tmp/frame-execution/connected-board-video-service-probe58-color-exposure'
    names=['guarded_async_fifo.sv','raw_axi_beat_unpacker.sv','raw_line_bank_ram.sv',
           'raw_line_feeder.sv','raw32_validated_packer.sv','raw_stream_ingress.sv']
    files=[base/'assembly'/n for n in names]+[base/'rtl/frame_buffer_V3/DC_FIFO.v']
    snapshots=[]
    for f in files:
        out=root/f.name;shutil.copy2(f,out)
        snapshots.append(dict(source=str(f),snapshot=str(out),sha256=sha(f)))
    template=HERE/'variants/raw_stream_test/tb_raw_stream_ingress.sv'
    s=template.read_text()
    changes=[('PPL=32,LINES=12','PPL=960,LINES=1080'),
             ('.FIFO_DEPTH(32)', '.FIFO_DEPTH(256)'),
             ('wire [4:0] ux;wire [3:0] uy','wire [9:0] ux;wire [10:0] uy'),
             ('#5.61 cam','#10 cam'),('#6.73 clk','#6.734 clk'),
             ('#3000000;', '#200000000;')]
    for before,after in changes:
        assert s.count(before)==1,before;s=s.replace(before,after)
    lo=s.index('  for(integer f=0;f<8;f=f+1)');hi=s.index('\n end\n initial begin #200000000',lo)
    s=s[:lo]+'''
  camera_frame(0,WORDS,1,3);wait_good(1);
  camera_frame(1,385440,0,3);wait_abort(1);
  if(!le || oe || aborts!=1)$fatal(1,"803_SHORT_CLASSIFICATION");
  camera_frame(2,WORDS,1,3);wait_good(2);
  if(cancels!=1 || dut.phase!=0 || request || ack || consumer_reset)
    $fatal(1,"803_RECOVERY_STUCK");
  $display("PASS RAW_803_FULL_GEOMETRY words=%0d short_words=385440 good=%0d cancel=%0d compared_pairs=%0d",WORDS,completed,cancels,checked);$finish;
'''+s[hi:]
    (root/'tb.sv').write_text(s)
    inputs=dict(inputs=snapshots,template=str(template),template_sha256=sha(template),
                runner_sha256=sha(__file__),tb_sha256=sha(root/'tb.sv'),
                scope='Frozen67/retained69 RAW ingress and line feeder, full1920x1080 geometry; simulated DMA cancellation drain only',
                simulation_host_timeout_seconds=180,hardware_access=False)
    (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
    os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    ms=Path('E:/quatra/questa_fse/win64');start=time.monotonic()
    try:
        run([ms/'vlib.exe','work'],root,'vlib.log',20)
        b,_=run([ms/'vlog.exe','-sv','-timescale','1ns/1ps',*[f.name for f in root.iterdir() if f.suffix in ('.v','.sv')]],root,'compile.log',30)
        assert 'Errors: 0, Warnings: 0' in b
        b,_=run([ms/'vsim.exe','-c','work.tb_raw_stream_ingress','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',180)
        assert 'PASS RAW_803_FULL_GEOMETRY' in b and 'Errors: 0, Warnings: 0' in b
        result=dict(status='PASS',line=re.search(r'^# PASS RAW_803.*$',b,re.M).group(0))
    except Exception as e:
        result=dict(status='FAIL',error=repr(e))
    result.update(seconds=time.monotonic()-start,inputs_sha256=sha(root/'inputs.json'),
                  hardware_access=False,whole_board=False,field_stall_root_cause_proved=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    return result['status']=='PASS'


if __name__=='__main__':raise SystemExit(0 if main() else 1)
