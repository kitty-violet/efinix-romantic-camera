"""Four-state producer/writer adjacency and cycle equivalence, frozen board source."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--board',type=Path,default=ROOT/'tmp/frame-execution/connected-board-native-lock1');a=p.parse_args()
 root=a.root.resolve();board=a.board.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 meta=json.loads((board/'inputs.json').read_text());indexed={Path(x['snapshot']).name:x for x in meta['inputs']}
 reference=Path(indexed['column_write8.sv']['snapshot']);expand=Path(indexed['result_pair_expand4.sv']['snapshot'])
 for f in (reference,expand):assert sha(f)==indexed[f.name]['sha256']
 candidate=HERE/'variants/paired_writer/column_write8.sv';tb=HERE/'tb_paired_writer_equivalence.sv'
 records=[]
 for src in (reference,expand,candidate,tb):
  dst=root/('column_write8_reference.sv' if src==reference else src.name)
  body=src.read_text()
  if src==reference:
   assert body.count('module column_write8(')==1;body=body.replace('module column_write8(','module column_write8_reference(',1)
  dst.write_text(body);records.append(dict(source=str(src),source_sha256=sha(src),snapshot=str(dst),sha256=sha(dst)))
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records,board_inputs_sha256=sha(board/'inputs.json'),runner_sha256=sha(__file__)),indent=2))
 run([MS/'vlib.exe','work'],root,'vlib.log')
 body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],root,'compile.log')
 assert 'Errors: 0, Warnings: 0' in body
 results=[]
 for seed in (71,391,1027):
  body,seconds=run([MS/'vsim.exe','-c','work.tb_paired_writer_equivalence',f'-gSEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,f'seed{seed}.log',timeout=90)
  lines=re.findall(r'^# (PASS PAIRED_WRITER.*)$',body,re.M)
  assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
  results.append(dict(seed=seed,line=lines[0],seconds=seconds,log_sha256=sha(root/f'seed{seed}.log')))
 for row in records:assert sha(row['source'])==row['source_sha256'] and sha(row['snapshot'])==row['sha256']
 result=dict(status='PAIRED_WRITER_FOUR_STATE_EQUIVALENCE_PASS',results=results,inputs_sha256=sha(root/'inputs.json'),whole_board=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
