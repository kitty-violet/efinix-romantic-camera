"""Graft the service changes into the actual frozen native design.

Replaces explicit state cones, not the stale assembly. Every changed primitive
and cut signal is recorded; encrypted blocks and retained primitives stay exact.
This preparation alone is not integration verification or board release.
"""
import argparse,collections,json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from video_service_graph import load,SOURCE
from native_local_drivers import CELL,render_token,lexical_tokens,rewrite_expression
from camera_joint_native import OUT
from complete_native_logic_split import protected_blocks
from video_service_display_native import extension as display_extension,explicit_net_assignments,split_net_declarations
from video_service_modes_native import extension as modes_extension
from video_service_native_binding import bind as bind_native,native_reads

VARIANT=HERE/'variants/video_service'
P='video_processing_pipeline/pipeline/'
RESET='u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn'
def net(n,b=None):return ('net',n,b)
def v(n,b=None):return render_token(net(n,b))

def boot_expression(q):
 name=q[1];bit=q[2]
 if name.startswith('cpu/sequencer/'):name=name.removeprefix('cpu/sequencer/')
 elif name.startswith('preview_boot/'):name=name.removeprefix('preview_boot/')
 else:
  name={'cpu/boot_observer_state':'state','cpu/boot_fault':'fault','cpu/boot_failed':'failed',
   'cpu/boot_terminal':'terminal','cpu/permit':'run_allowed','preview_frame_budget':'frames'}[name]
 assert name in 'tag state operation_valid run_ticks operation_state frames low_word nibble index bank waiting fault row high_word sample crc base_count bus_ticks run_allowed terminal failed diag_polls'.split(),name
 return name+(f'[{bit}]' if bit is not None else '')

def owner_expression(q):
 n,b=q[1:];prefix=P+'owner/'
 if n in ('committed_frames','dropped_results'):return f'{n}[{b}]'
 if n=='bringup_pipeline_status':assert b==0;return 'error'
 if n=='video_processing_pipeline/frame_states':return f'state[{b//3}][{b%3}]'
 if n in (P+'compute_input',P+'display_input'):
  assert b in (22,23);return ('c_candidate' if 'compute' in n else 'd_candidate')+f'[{b-22}]'
 if n.startswith('i757/'+prefix+'order['):
  slot=int(re.search(r'order\[(\d)\]',n)[1]);return f'(~rank[{slot}][{b}])' if b<2 else "1'b0"
 if n.startswith(prefix+'i551/frame['):return 'frame'+n.split('/frame')[1]+f'[{b}]'
 assert n.startswith(prefix),q
 n=n.removeprefix(prefix)
 if n=='next_order':return "1'b0"
 assert n in 'capturing computing displaying c_offer d_offer cap_slot comp_slot disp_slot c_candidate d_candidate'.split(),q
 return n+(f'[{b}]' if b is not None else '')

def expose(s,module,rows):
 # Export only original state outputs that actually feed retained logic.
 assert rows
 anchor=s.index(');',s.index('module '))
 s=s[:anchor]+f',\n output wire [{len(rows)-1}:0] service_native_state\n'+s[anchor:]
 return s.replace('endmodule',' assign service_native_state={'+', '.join(r['expression'] for r in reversed(rows))+'};\nendmodule')

def build(out):
 input_hashes={str(p):sha(p) for p in [SOURCE,Path(__file__),HERE/'video_service_native_binding.py',HERE/'video_service_display_native.py',HERE/'video_service_modes_native.py',*VARIANT.glob('*.sv')]}
 D=load();g,raw,ps,drivers,loads=[D[k] for k in ('g','raw','ps','drivers','loads')];s=SOURCE.read_text()
 assert sha(SOURCE)==D['sha256']
 boot=set();owner=set();admission=set();expressions={}
 for n,(k,p) in g.items():
  if k!='EFX_FF':continue
  q=p['Q'][0]
  if n.startswith('preview_admitted['):admission.add(n);continue
  if q[1].startswith(('preview_boot/','cpu/sequencer/')) or q[1] in ('cpu/boot_observer_state','cpu/boot_fault','cpu/boot_failed','cpu/boot_terminal','cpu/permit','preview_frame_budget'):
   # cpu/sequencer names are the explicitly retained sequencer outputs, not CPU state.
   expressions[q]=('boot',boot_expression(q));boot.add(n)
  elif any(t in q[1] for t in ('/owner/','video_processing_pipeline/frame_states')) or q[1] in ('committed_frames','dropped_results',P+'compute_input',P+'display_input') or q==net('bringup_pipeline_status',0):
   expressions[q]=('owner',owner_expression(q));owner.add(n)
 rom='preview_boot/operations';assert g[rom][0]=='EFX_RAM10';boot.add(rom)
 for i,q in enumerate(g[rom][1]['RDATA']):
  if i>=2:expressions[q]=('boot',f'operation[{15-i}]')
 assert len(boot)==296 and len(owner)==263 and len(admission)==10,(len(boot),len(owner),len(admission))
 forced=boot|owner|admission
 # The admission and the two frame-start handshakes have new owners.
 supplied={net('cpu/boot_valid'):'service_boot_valid',
  net('video_processing_pipeline/compute_start'):'service_compute_fire',
  net(P+'display/start_fire'):'service_display_fire',
  net(P+'capture/start_fire'):v('camera_burst_capture/start_fire')}
 supplied.update({net('preview_admitted',i):f'service_admitted[{i}]' for i in range(10)})
 for q in supplied:forced.add(drivers[q])
 # Keep command/reply, reset, and new wrapper inputs outside the cut.
 boundary={net(n) for n in ['process_clk_150',RESET,'w_arstn','ceg_net4871','cpu/boot_ready','n78042_2',
  'cpu/reply_error','cpu/fatal_bus_error','fixed_boot_run_allowed','cal_done_sync','control_enable',
  'cnn_start_request','camera_burst_capture/start_fire','cap_ready','cap_failed',P+'prod_frame_valid',P+'compute_active']}
 boundary.update(net('cpu/reply_data',i) for i in range(32))
 removed=set(forced);todo=list(forced)
 while todo:
  n=todo.pop()
  for pin,ts in g[n][1].items():
   if pin in OUT[g[n][0]] or pin in ('CLK','SR'):continue
   for q in ts:
    if q in boundary:continue
    parent=drivers.get(q)
    if parent and parent not in removed and g[parent][0] in ('EFX_LUT4','EFX_ADD'):
     removed.add(parent);todo.append(parent)
 # Shared combinational logic cannot be removed. Repeatedly retain anything
 # that feeds outside, except explicit replacement output drivers.
 while True:
  keep={n for n in removed-forced for pin,ts in g[n][1].items() if pin in OUT[g[n][0]]
   for q in ts if q in boundary or any(cell not in removed for cell,_ in loads[q])}
  if not keep:break
  removed-=keep
 rows={'boot':[],'owner':[]}
 for n in sorted(boot|owner):
  for pin,ts in g[n][1].items():
   if pin not in OUT[g[n][0]]:continue
   for q in ts:
    if any(cell not in removed for cell,_ in loads[q]):
     group,expr=expressions[q];rows[group].append(dict(cell=n,token=q,expression=expr))
 outputs=set(supplied)|{tuple(r['token']) for rr in rows.values() for r in rr}
 for n in removed:
  for pin,ts in g[n][1].items():
   if pin not in OUT[g[n][0]]:continue
   for q in ts:
    assert q in outputs or all(cell in removed for cell,_ in loads[q]),('cut leak',n,pin,q)
 # Only the upstream scaler sees the discard-ready path. The writer retains
 # its real ready. All uses of source-valid in the writer are gate-controlled.
 edits={};gate_valid=net('video_processing_pipeline/down_valid');ready=net('cap_ready')
 for n,pin in loads[gate_valid]:
  # The upstream transfer predicate and raw-source observation must still
  # consume/count a discarded frame. Only writer-valid is suppressed.
  if n not in removed and n not in ('LUT__153476','LUT__243373'):
   edits.setdefault(n,{}).setdefault(pin,{})[gate_valid]=net('service_capture_valid')
 for n,pin in loads[ready]:
  if n not in removed:
   # We audit each load in the manifest. PASS forwards the same readiness;
   # DROP has no writer valid, so internal input_fire cannot be fabricated.
   edits.setdefault(n,{}).setdefault(pin,{})[ready]=net('service_source_ready')
 # Shared compute request gate now includes draining already captured frames.
 enable=drivers[net('n63096_2')];assert ps[enable]['LUTMASK']==(16,0x4000)
 edits.setdefault(enable,{})['I3']={net('fixed_boot_run_allowed'):net('service_compute_permit')}
 assert enable not in removed
 display_code,display_receipt=display_extension(D,removed,edits)
 modes_code,modes_receipt=modes_extension(D,removed,edits)
 def change(m):
  n=m[2].lstrip('\\')
  if n in removed:return ''
  if n not in edits:return m[0]
  ports=re.sub(r'\.(\w+)\(([^()]*)\)',lambda p:'.'+p[1]+'('+rewrite_expression(p[2],edits[n].get(p[1],{}))+')',m[3])
  return m[0][:m.start(3)-m.start()]+ports+m[0][m.end(3)-m.start():]
 body=CELL.sub(change,s)
 body=re.sub(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',lambda m:'' if m[1].lstrip('\\') in removed else m[0],body,flags=re.M)
 code=r'''
    wire service_boot_valid,service_prepared,service_draining,service_button_event;
    wire service_compute_fire,service_display_fire,service_capture_valid,service_source_ready;
    wire service_owner_compute_valid,service_owner_display_valid,service_owner_capture_ready;
    wire service_front_valid,service_passing; wire [1:0] service_front_bank;
    wire [31:0] service_repeated,service_capture_admitted,service_capture_skipped,service_capture_orphans;
    wire [31:0] service_rejected_buttons,service_compute_admitted;
    wire [9:0] service_frame_budget,service_admitted;
    wire service_compute_permit=(fixed_boot_run_allowed || service_draining) && !service_bypass_compute;
    wire service_reset=!RESET;
    wire service_work_drained=!cap_active && !service_passing && !COMPUTING &&
     SLOTS[2:0]!=1 && SLOTS[5:3]!=1 && SLOTS[8:6]!=1 &&
     SLOTS[2:0]!=2 && SLOTS[5:3]!=2 && SLOTS[8:6]!=2 &&
     SLOTS[2:0]!=3 && SLOTS[5:3]!=3 && SLOTS[8:6]!=3;
    wire service_host_write=ceg_net4871 && HOST_IDLE && HOST_CHOOSE && HOST_WRITE;
    wire service_payload_changed=service_inspect_changed || (service_host_write &&
      ((HOST_ADDRESS>=32'he1000000 && HOST_ADDRESS<32'he2000000) ||
       (HOST_ADDRESS>=32'hf9000f08 && HOST_ADDRESS<=32'hf9000f1c) ||
       (HOST_ADDRESS>=32'hf8100010 && HOST_ADDRESS<=32'hf8100054)));
    video_run_button service_button(.clk(process_clk_150),.reset(!ceg_net4871),
      .key_n(service_key_n),.press(service_button_event));
    cnn_preview_boot service_boot(.clk(process_clk_150),.reset(!ceg_net4871),
      .button_event(service_button_event || service_auto_stop),.payload_changed(service_payload_changed),.work_drained(service_work_drained),
      .prepared(service_prepared),.draining(service_draining),.rejected_buttons(service_rejected_buttons),
      .cmd_valid(service_boot_valid),.cmd_write(),.cmd_address(),.cmd_data(),.cmd_ready(BOOT_READY),
      .rsp_valid(n78042_2),.rsp_error(REPLY_ERROR || FATAL_BUS),.rsp_data(REPLY_DATA),
      .run_allowed(),.terminal(),.failed(),.fault(),.observed_crc(),.observer_state(),
      .native_state(),.frame_budget(service_frame_budget),.service_native_state(BOOT_EXPORTS));
    cnn_frame_admission service_admission(.clk(process_clk_150),.reset(!w_arstn),
      .permit(service_compute_permit),.start_request(cnn_start_request),.requested_frames(service_frame_budget),
      .start_accept(service_compute_fire),.admitted(service_admitted),.lifetime_admitted(service_compute_admitted));
    frame_pair_owner service_owner(.clk(process_clk_150),.reset(service_reset),.bypass_compute(service_bypass_compute),
      .capture_valid(CAP_START),.capture_ready(service_owner_capture_ready),.capture_frame_id(service_selected_id),
      .capture_bank(),.capture_address(),.capture_done(cap_done),.capture_failed(cap_failed),
      .compute_valid(service_owner_compute_valid),.compute_ready(service_compute_fire),
      .compute_frame_id(),.compute_input_address(),.compute_output_address(),
      .compute_done(COMPUTING && PROD_DONE && !bringup_pipeline_status[2]),
      .compute_failed(COMPUTING && bringup_pipeline_status[2]),
      .display_valid(service_owner_display_valid),.display_ready(service_display_fire),
      .display_frame_id(),.display_input_address(),.display_output_address(),.display_done(DISPLAY_DONE),
      .error(),.states(),.committed_frames(),.dropped_results(),.repeated_reads(service_repeated),
      .front_valid(service_front_valid),.front_bank(service_front_bank),.service_native_state(OWNER_EXPORTS));
    assign service_display_fire=service_owner_display_valid && !service_reset &&
     cal_done_sync && !STOPPED && !bringup_pipeline_status[3] && DISPLAY_STATE==0 && !service_video_inhibit;
    rgb_frame_admission service_capture_gate(.clk(process_clk_150),.reset(service_reset),.abort_frame(service_capture_abort),
      .allow_capture(service_allow_capture),.slot_ready(service_owner_capture_ready),
      .s_valid(service_selected_valid),.s_ready(service_gate_ready),.s_rgb(service_selected_rgb),.s_frame_id(service_selected_id),
      .s_sof(service_selected_sof),.s_eol(service_selected_eol),.s_eof(service_selected_eof),
      .m_valid(service_capture_valid),.m_ready(cap_ready),.m_rgb(),.m_frame_id(),.m_sof(),.m_eol(),.m_eof(),
      .admitted_frames(service_capture_admitted),.skipped_frames(service_capture_skipped),
      .orphan_pixels(service_capture_orphans),.passing(service_passing));
'''
 replacements={'RESET':RESET,'COMPUTING':P+'compute_active','SLOTS':'video_processing_pipeline/frame_states',
  'HOST_IDLE':'cpu/ctrl_state_match','HOST_CHOOSE':'cpu/choose_host','HOST_WRITE':'cpu/host_write','HOST_ADDRESS':'cpu/host_address',
  'BOOT_READY':'cpu/boot_ready','REPLY_ERROR':'cpu/reply_error','FATAL_BUS':'cpu/fatal_bus_error','REPLY_DATA':'cpu/reply_data',
  'CAP_START':'camera_burst_capture/start_fire','DOWN_ID':'video_processing_pipeline/down_frame_id','PROD_DONE':P+'prod_frame_valid',
  'DISPLAY_DONE':'video_processing_pipeline/display_done','STOPPED':P+'stopped','DISPLAY_STATE':P+'display/state',
  'DOWN_VALID':'video_processing_pipeline/down_valid','DOWN_RGB':'video_processing_pipeline/down_rgb',
  'DOWN_SOF':'video_processing_pipeline/downscaler/out_sof','DOWN_EOL':'video_processing_pipeline/downscaler/out_eol','DOWN_EOF':'video_processing_pipeline/downscaler/out_eof'}
 code=code.replace('HOST_IDLE','HOST_IDLE[0]')
 code=re.sub(r'\b('+ '|'.join(replacements)+r')\b',lambda m:v(replacements[m[0]]),code)
 for group in rows:
  code=code.replace(group.upper()+'_EXPORTS','{'+', '.join(render_token(tuple(r['token'])) for r in reversed(rows[group]))+'}')
 for q,expr in supplied.items():code+='\n    assign '+render_token(q)+'='+expr+';'
 code+='\n'+display_code+'\n'+modes_code
 code,binding_receipt=bind_native(code,D)
 binding_receipt.update(native_reads(code,s,D,outputs|set(supplied),removed))
 binding_receipt['observation_counter_contract']={
  'display_frame_counters_modulus':65536,'pixel_mailbox_sequence_modulus':65536,
  'modulo_16bit_fields':['capture_admitted','capture_skipped','owner_dropped_results','owner_repeated_reads'],
  'saturating_16bit_fields':['orphan_pixels','rejected_mode_changes','source_area_frames','pixel_cache_timeouts','recoveries','compute_admitted','rejected_buttons'],
  'commit_stall_saturates_at_cycles':300000000,
  'timestamps_bits':32,'ai_committed_frames_bits':32,
  'first_fault_and_reset_history_retained':True}
 # Efinity 2026.1 drops a net-declaration assignment when the net was
 # implicitly referenced earlier by a native primitive. Explicit continuous
 # assignments are required at this late graft point (repro wire-init1).
 code=explicit_net_assignments(code)
 declarations,code=split_net_declarations(code)
 # Declare every new native net before its first primitive use. This also
 # avoids implicit one-bit nets before vector port connections in old vlog.
 first_cell=CELL.search(body).start()
 body=body[:first_cell]+'\n'+'\n'.join(declarations)+'\n'+body[first_cell:]
 # Add a separate known pushbutton input; all existing pins remain fixed.
 body=body.replace('module ti60f225_oob_top (','module ti60f225_oob_top (service_key_n, ',1)
 idx=body.index('    input jtag_inst1_CAPTURE');body=body[:idx]+'    input service_key_n;\n'+body[idx:]
 anchor=body.index('    protected_cell protected_inst (');body=body[:anchor]+code+'\n'+body[anchor:]
 assert protected_blocks(body)==protected_blocks(s)
 assert all(sha(p)==h for p,h in input_hashes.items()),'source changed during native preparation'
 assert not out.exists();out.mkdir(parents=True)
 (out/'native_source.v').write_text(body,encoding='utf8')
 for group,filename,module in [('boot','cnn_preview_boot.sv','cnn_preview_boot'),('owner','frame_pair_owner.sv','frame_pair_owner')]:
  (out/filename).write_text(expose((VARIANT/filename).read_text(),module,rows[group]),encoding='utf8')
 for filename in ['cnn_frame_admission.sv','rgb_frame_admission.sv','video_run_button.sv','video_display_observer.sv','video_debug_mailbox.sv','video_display_recovery.sv','video_service_modes.sv','video_service_health.sv','video_dualclock_sample.sv']:
  (out/filename).write_bytes((VARIANT/filename).read_bytes())
 report=dict(status='SERVICE_NATIVE_CANDIDATE_NOT_YET_INTEGRATION_VERIFIED',source=str(SOURCE),source_sha256=sha(SOURCE),
  removed=sorted(removed),removed_kinds=dict(collections.Counter(g[n][0] for n in removed)),exports=rows,
  changed_loads=[dict(cell=n,pin=p,from_token=f,to_token=t) for n,pins in edits.items() for p,aliases in pins.items() for f,t in aliases.items()],
  supplied=[dict(token=q,expression=e) for q,e in supplied.items()],protected_blocks_unchanged=True,
  retained_primitive_count=len(g)-len(removed),new_button='SW0 C4/GPIOL_07; needs peripheral/CDC constraint integration',
  hardware_access=False,board_release=False,physical_pass=False,inputs=input_hashes,native_binding=binding_receipt,display_extension=display_receipt,modes_extension=modes_receipt)
 (out/'graft.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 print(json.dumps({k:v for k,v in report.items() if k not in ('removed','exports','changed_loads','supplied','inputs','display_extension','serializer_extension')},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp');build(out)
