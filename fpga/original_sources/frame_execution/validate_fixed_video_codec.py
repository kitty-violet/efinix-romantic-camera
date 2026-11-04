"""Retain the actual vendor pixel encoder; A/B its complete 16-row output."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
from validate_board_dvi import TB,decode

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);a=p.parse_args()
 r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
 proof=json.loads((a.proof/'result.json').read_text());assert proof['status']=='FIXED_VIDEO_AUX_PROTOCOL_PASS'
 aux=HERE/'variants/fixed_video_aux/tx_aux_video_only.sv';meta=json.loads((a.proof/'inputs.json').read_text())
 assert sha(a.proof/'inputs.json')==proof['inputs_sha256'] and sha(a.proof/'simulation.log')==proof['log_sha256']
 assert next(x['sha256'] for x in meta['inputs'] if Path(x['snapshot']).name==aux.name)==sha(aux)
 records=[];runs=[];outputs=[]
 try:
  for candidate in (False,True):
   d=r/('candidate' if candidate else 'reference');d.mkdir()
   base=ROOT/'tmp/frame-execution/connected-board-raw10-only2/rtl/dvi_tx'
   files=[f for f in base.iterdir() if f.suffix in ('.v','.sv','.vh')]+[EF/'sim_models/verilog/efx_srl8.v']
   snapshots=[]
   for f in files:
    dest=d/f.name
    if candidate and f.name=='tx_aux.v_encrypted.v':
     f=aux;dest.write_text(aux.read_text().replace('module tx_aux_video_only(','module tx_aux(',1))
    else:shutil.copy2(f,dest)
    records.append(dict(source=str(f),snapshot=str(dest),source_sha256=sha(f),sha256=sha(dest)));snapshots.append(dest)
   (d/'tb_board_dvi.sv').write_text(TB)
   records.append(dict(snapshot=str(d/'tb_board_dvi.sv'),sha256=sha(d/'tb_board_dvi.sv')))
   run([MS/'vlib.exe','work'],d,'vlib.log')
   run([MS/'vlog.exe','-sv','-timescale','1ns/1ps','+incdir+.',*[f for f in snapshots if f.suffix!='.vh'],d/'tb_board_dvi.sv'],d,'compile.log')
   body,sec=run([MS/'vsim.exe','-c','work.tb_board_dvi','-do','onerror {quit -code 1}; run -all; quit -code 0'],d,'simulation.log',timeout=120)
   assert 'PASS DVI_STIMULUS_COMPLETE' in body and 'Errors: 0, Warnings: 0' in body
   rows=[line.split() for line in (d/'symbols.txt').read_text().splitlines()];assert len(rows)==40000
   valid=[i for i,row in enumerate(rows) if row[1]=='1'];assert len(valid)==30720
   def matches(delay):
    for i in valid:
     symbols=rows[i+delay][3:6]
     if any(re.search('[xz]',s,re.I) for s in symbols):return False
     if sum(decode(s)<<(8*k) for k,s in enumerate(symbols))!=int(rows[i][2],16):return False
    return True
   delays=[i for i in range(65) if matches(i)];assert len(delays)==1,delays
   outputs.append([rows[i+delays[0]][3:7] for i in valid])
   runs.append(dict(candidate=candidate,latency=delays[0],pixels=len(valid),seconds=sec,log_sha256=sha(d/'simulation.log'),symbols_sha256=sha(d/'symbols.txt')))
  assert runs[0]['latency']==runs[1]['latency'] and outputs[0]==outputs[1]
  for x in records:
   assert sha(x['snapshot'])==x['sha256']
   if 'source' in x:assert sha(x['source'])==x['source_sha256']
  (r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),protocol_proof=str(a.proof),protocol_result_sha256=sha(a.proof/'result.json')),indent=2))
  result=dict(status='FIXED_VIDEO_CODEC_AB_PASS',runs=runs,pixel_tmds_bit_exact=True,inputs_sha256=sha(r/'inputs.json'),whole_screen=False,phy=False,board_release=False)
  (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
