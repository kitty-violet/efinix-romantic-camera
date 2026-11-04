"""Bound native delta: finite deadline, correct readback, admission and progress.

The protected PHY and all arithmetic/data movers remain byte-identical. This
is a candidate generator, never a release or a hardware download command.
"""
import json,re,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from native_local_drivers import CELL,graph,raw_cells,render_token
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
PARENT=ROOT/'tmp/frame-execution/connected-board-camera-factor-compact1'
OUT=ROOT/'tmp/frame-execution/cnn-run-budget-native4'
P='video_processing_pipeline/pipeline/cnn/'
TIMEOUT={'LUT__132827':'I1','LUT__133011':'I1','LUT__150477':'I0','LUT__150650':'I0','LUT__150651':'I2'}
def named(s):return '\\'+s+' ' if any(c in s for c in '/[]~') else s
def cell(kind,name,pins,ps):
 s='    '+kind+' '+named(name)+' ('+', '.join('.'+p+'('+v+')' for p,v in pins.items())+') /* cnn_run_budget native primitive */ ;\n'
 return s+''.join('    defparam '+named(name)+'.'+k+' = '+str(w)+"'d"+str(v)+';\n' for k,(w,v) in ps.items())
def main():
 assert not OUT.exists();OUT.mkdir()
 checks=[ROOT/'tmp/frame-execution'/n/'result.json' for n in ['cnn-run-budget-short3','cnn-owner-admission1','cnn-budget-real-program1']]
 assert [json.loads(p.read_text())['status'] for p in checks]==['CNN_RUN_BUDGET_SHORT_PASS','OWNER_ADMISSION_RACE_REPRODUCED_AND_FIXED','ACCEPTED_A_AUTO_DW_BUDGET_ISSUE_PASS']
 src=PARENT/'outflow/ti60f225_oob.map.v';s=src.read_text();raw=raw_cells(s);g=graph(s);ps=params(s)
 survey=json.loads((ROOT/'tmp/board-test/cnn-budget-progress-survey1.json').read_text());assert survey['map_sha256']==sha(src)
 edits={n:{pin:'cnn_cycles_hi[1]'} for n,pin in TIMEOUT.items()}
 for n,p in TIMEOUT.items():assert raw[n][1][p]=='production_cycles[23]'
 add=P+'issue/add_502/i23';assert 'CO' not in raw[add][1];edits[add]={'CO':'cnn_cycles_carry24'}
 # Preserve original decode/qualification. Insert one extra NAND term into
 # each existing inverted read-data sum, rather than overriding CPU replies.
 for n,old,new in [('LUT__162230','n78804_2','cnn_old_read24'),('LUT__162241','n78814_2','cnn_old_read25'),
                   ('LUT__132753','\\video_processing_pipeline/compute_start ','cnn_start_request')]:
  assert raw[n][1]['O']==old;edits[n]={'O':new}
 slots=json.loads((PARENT/'camera_graft.json').read_text())['diagnostic_slots'];assert len(set(slots))==111
 snap=['cpu/system_hardJtag_debug_logic_jtagBridge/observation/'+f'held_payload[{832+i}]~FF' for i in slots]
 old_page=[]
 for i,n in enumerate(snap):old_page.append(raw[n][1]['D']);edits[n]={'D':f'cnn_observation_page[{i}]'}
 def change(m):
  n=m[2].lstrip('\\')
  if n not in edits:return m[0]
  body=m[0]
  for pin,value in edits[n].items():
   if pin=='CO':
    assert n==add
    body,count=re.subn(r'\.O\(([^()]*)\)',lambda z:z[0]+', .CO('+value+')',body);assert count==1
   else:
    body,count=re.subn(r'\.'+pin+r'\([^()]*\)',lambda z:'.'+pin+'('+value+')',body);assert count==1
  return body
 body=CELL.sub(change,s)
 extra='''
    wire [1:0] cnn_cycles_hi,cnn_cycles_next;
    wire cnn_cycles_carry24,cnn_cycles_carry25;
    wire cnn_old_read24,cnn_old_read25,cnn_start_request;
    wire [4:0] cnn_admitted;
    reg cnn_run_seen;
    always @(posedge process_clk_150) begin
      if(!w_arstn) cnn_run_seen<=0;
      else if(\\video_processing_pipeline/compute_start ) cnn_run_seen<=1;
    end
    cnn_frame_admission cnn_run_admission(
      .clk(process_clk_150),.reset(!w_arstn),.permit(fixed_boot_run_allowed),
      .requested_frames(\\cpu/sequencer/frames ),.start_request(cnn_start_request),
      .start_accept(\\video_processing_pipeline/compute_start ),.admitted(cnn_admitted));
    wire [110:0] cnn_original_page={OLD_PAGE};
    // D-page is only selected after a real frame start, on the old stream
    // phase. The alternate C7 configuration page and global diagnostics stay.
    wire [110:0] cnn_progress_page={
      \\video_processing_pipeline/pipeline/cnn/issue_error ,
      \\video_processing_pipeline/pipeline/cnn/fetch_verified ,
      \\video_processing_pipeline/pipeline/cnn/issue_busy ,
      control_enable,cnn_admitted,\\video_processing_pipeline/frame_states ,
      \\video_processing_pipeline/pipeline/cnn/nodes_completed ,4'hd,
      \\video_processing_pipeline/pipeline/cnn/issue/dwait ,
      \\video_processing_pipeline/pipeline/cnn/issue/hstate ,
      \\video_processing_pipeline/pipeline/cnn/issue/cstate ,
      \\video_processing_pipeline/pipeline/cnn/dma_completed ,
      \\video_processing_pipeline/pipeline/cnn/compute_completed ,cnn_cycles_hi,production_cycles[23:0]};
    wire [110:0] cnn_observation_page=
      (cnn_run_seen && \\cpu/system_hardJtag_debug_logic_jtagBridge/observation/request_sync ) ? cnn_progress_page:cnn_original_page;
'''.replace('OLD_PAGE',', '.join(reversed(old_page)))
 for i in range(2):
  pins=dict(raw['production_cycles[23]~FF'][1]);pins['D']=f'cnn_cycles_next[{i}]';pins['Q']=f'cnn_cycles_hi[{i}]'
  extra+=cell('EFX_FF',f'CNN_RUN_CYCLE_FF{i}',pins,ps['production_cycles[23]~FF'])
  pins=dict(I0=f'cnn_cycles_hi[{i}]',I1="1'b0",CI=f'cnn_cycles_carry{24+i}',O=f'cnn_cycles_next[{i}]')
  if i==0:pins['CO']='cnn_cycles_carry25'
  extra+=cell('EFX_ADD',f'CNN_RUN_CYCLE_ADD{i}',pins,ps.get(add,{}))
 for i,(old,out) in enumerate([('cnn_old_read24','n78804_2'),('cnn_old_read25','n78814_2')]):
  # O=I0 & !(I1&I2), independent of unused I3.
  extra+=cell('EFX_LUT4',f'CNN_RUN_READ_HIGH{i}',dict(I0=old,I1=f'cnn_cycles_hi[{i}]',I2='n72363_2',O=out),dict(LUTMASK=(16,0x2a2a)))
 anchor=body.index('    protected_cell protected_inst (');body=body[:anchor]+extra+body[anchor:]
 ng=graph(body);np=params(body);added=set(ng)-set(g)
 assert len(added)==6 and not(set(g)-set(ng)),(sorted(added),sorted(set(g)-set(ng)))
 for n in g:
  assert ps.get(n)==np.get(n),n
  if n not in edits:assert g[n]==ng[n],n
  else:
   for p,ts in g[n][1].items():
    if p not in edits[n]:assert ts==ng[n][1][p],(n,p)
 assert protected_blocks(s)==protected_blocks(body)
 # Added diagnostic inputs must be existing driven scalar bits. Unlike a
 # source alias, these names refer to actual FF outputs in the selected map.
 outputs={'Q','O','CO','RDATA'}
 driven={t for k,pins in g.values() for pin,ts in pins.items() if pin in outputs for t in ts}
 taps=[('net',P+n,i) for n,w in [('compute_completed',14),('dma_completed',14),('nodes_completed',29),('issue/cstate',3),('issue/hstate',2)] for i in range(w)]
 taps += [('net','video_processing_pipeline/frame_states',i) for i in range(9)]
 taps += [('net',P+n,None) for n in ['issue/dwait','issue_busy','issue_error','fetch_verified']]
 taps += [('net','cpu/sequencer/frames',i) for i in range(5)]
 assert all(t in driven for t in taps),[t for t in taps if t not in driven]
 (OUT/'native_source.v').write_text(body,encoding='utf8')
 evidence={str(f):sha(f) for f in [src,Path(__file__),HERE/'variants/run_budget/cnn_frame_admission.sv',*checks]}
 result=dict(status='CNN_RUN_BUDGET_NATIVE_DELTA_PREPARED',source_sha256=sha(OUT/'native_source.v'),
  parent_map_sha256=sha(src),edits=edits,added_cells=sorted(added),counter_limit=33554432,
  protected_unchanged=True,unmodified_cells=len(g)-len(edits),inputs=evidence,
  diagnostic_schema='D-run-progress-111',physical_pass=False,hardware_access=False)
 (OUT/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],len(edits),len(added),flush=True)
if __name__=='__main__':main()
