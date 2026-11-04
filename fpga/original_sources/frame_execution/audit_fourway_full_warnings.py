"""Review every warning in the bound continuous-frame native snapshot."""
import argparse,collections,json,re
from pathlib import Path
from run_units import sha
POLICY={
 'efx_dsp24.v':'Official primitive: signed/unsigned widening and string parameter comparisons; source unchanged.',
 'efx_ram10.v':'Official primitive: string literals have unequal packed widths; source unchanged.',
 'efx_dpram10.v':'Official primitive: string mode comparisons; source unchanged.',
 'shared_rq_oc8_cluster.sv':'Unsigned two-bit group compared with integer loop 0..3; zero-extension preserves value.',
 'finish_rr_arbiter.sv':'Unsigned owner compared with nonnegative NUM_FRONTS-1 (7); width extension only.',
 'pair_column_reader.sv':'Unsigned eleven-bit local x shifted by two then promoted to nonnegative int; address underflow/1024 guard precedes bus offer.',
 'compute_column_reader.sv':'Unsigned thirteen-bit packet index widened to fourteen before bounded group shift; no lost upper bits.',
 'shared_pool_banked_fifo.sv':'Unsigned three-bit selected owner compared with loop 0..7; equal numeric interpretation.',
 'frame_arena_row_dma.sv':'4KiB boundary beats 1..256 compared against unsigned bounded burst <=128; nine-bit selected length preserves value.',
 'next_header_prefetch.sv':'Unsigned field counts/index promoted to int in byte-lane selection; groups1/2/4 and offsets guarded.',
 'compute_column_cluster.sv':'Feed width widened in packet-count expressions; pixels1..320/stride1..160, no truncation before count calculation.',
 'pair_column_dispatch32.sv':'Generated DP=0 instance makes unsigned phase>=0 constant true, intentionally age=phase; other DP1..8 remain variable.'
}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert not a.output.exists()
 m=json.loads((a.root/'inputs.json').read_text());src={Path(r['snapshot']).name:r for r in m['inputs']}
 text=(a.root/'compile.log').read_text();warn=re.findall(r'^%Warning-([A-Z0-9_]+): (.+?):(\d+):(\d+): (.*)$',text,re.M)
 assert warn and not re.search(r'%Error',text);rows=[]
 for code,path,line,col,msg in warn:
  name=Path(path.replace('\\','/')).name
  assert code in ('WIDTHEXPAND','UNSIGNED') and name in POLICY,(code,name,line,msg)
  assert code!='UNSIGNED' or name=='pair_column_dispatch32.sv'
  r=src[name];assert sha(r['snapshot'])==r['sha256']
  source=Path(r['snapshot']).read_text().splitlines()[int(line)-1]
  rows.append(dict(code=code,file=name,line=int(line),source=source,message=msg,review=POLICY[name],source_sha256=r['sha256']))
 result=dict(status='BOUND_NATIVE_WIDTH_WARNINGS_REVIEWED',warnings=len(rows),codes=dict(collections.Counter(r['code'] for r in rows)),rows=rows,
  no_latch_loop_range_implicit=True,physical_timing=False,inputs_sha256=sha(a.root/'inputs.json'),compile_log_sha256=sha(a.root/'compile.log'))
 a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','warnings','codes')}))
if __name__=='__main__':main()

