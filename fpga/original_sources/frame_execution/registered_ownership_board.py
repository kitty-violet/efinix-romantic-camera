"""Preserve the full board at 150 MHz; register aux and RQ job selection.

Generate an independent candidate. Immutable parent proofs are not rewritten.
The new ready latency must be checked by transaction-based short tests.
"""
import argparse,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from prepare_ranged_control_board import audit_delta as audit_parent
from audit_control_state_map import audit as audit_state
from audit_snapshot_cells import audit as audit_sync
from windows_process_tree import stop_launched_tree

BASE=ROOT/'tmp/frame-execution/connected-board-ranged-controls1'
CHANGED=('row_pair_aux_router.sv','fixed_rq32_mixed.sv')

def once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

def router(text):
    lo=text.index('  reg [2:0] select;');hi=text.index('  // Compare client addresses',lo)
    text=text[:lo]+'''  reg [2:0] choice;
  always @*begin
   choice=0;
   case(next_owner[b])
    0:if(valid[0])choice=3'b001;else if(valid[1])choice=3'b010;else if(valid[2])choice=3'b100;
    1:if(valid[1])choice=3'b010;else if(valid[2])choice=3'b100;else if(valid[0])choice=3'b001;
    default:if(valid[2])choice=3'b100;else if(valid[0])choice=3'b001;else if(valid[1])choice=3'b010;
   endcase
  end
  // Reserve an owner before exposing its payload to RAM. Late cancellation
  // still suppresses aux_valid on this same cycle, but cannot steer WDATAB.
  // The producer retains its request until the actual RAM handshake.
  wire [2:0] select={held[b] && held_owner[b]==2,
                     held[b] && held_owner[b]==1,
                     held[b] && held_owner[b]==0};
''' + text[hi:]
    text=once(text,'  wire have=|select;\n  wire [1:0] grant={select[2],select[1]};',
        '  wire have=|(select & valid);\n  wire [1:0] grant=held_owner[b];')
    text=once(text,'    if(aux_valid[b] && !aux_ready[b])begin held[b]<=1;held_owner[b]<=grant;end',
        '''    if(!held[b] && (|choice))begin
     held[b]<=1;held_owner[b]<=choice[2] ? 2'd2:choice[1] ? 2'd1:2'd0;
    end''')
    return text

def rq(text):
    text=once(text,' reg preparing;reg [1:0] correction_wait;',
        ''' reg preparing;reg [1:0] correction_wait;
 // Job selection is a separate clock boundary. Payload remains with its
 // producer until in_ready, so no second wide data buffer is required.
 reg selected;reg [1:0] selected_kind;reg [31:0] selected_split;
 wire [31:0] split_lane;wire issue;
 always @(posedge clk)begin
  if(reset || cancel)begin selected<=0;selected_kind<=0;selected_split<=0;end
  else if(active)begin
   if(!selected && (|eligible))begin
    selected<=1;selected_kind<=eligible[2] ? 2'd2:eligible[1] ? 2'd1:2'd0;
    selected_split<=split_lane;
   end
   if(issue)selected<=0;
  end
 end''')
    text=once(text,'assign split_channel[c]=split_lane[c] || split_lane[8+c] || split_lane[16+c] || split_lane[24+c];',
        'assign split_channel[c]=selected_split[c] || selected_split[8+c] || selected_split[16+c] || selected_split[24+c];')
    text=once(text," wire [1:0] chosen=preparing ? 2'd0:eligible[2] ? 2'd2:eligible[1] ? 2'd1:2'd0;\n wire need_correction=chosen==0 && |split_lane;\n wire start_correction=(|eligible) && need_correction && !preparing;",
        ''' wire [1:0] chosen=selected_kind;
 wire selected_eligible=selected && eligible[chosen];
 wire need_correction=chosen==0 && |selected_split;
 // Data steering uses only reserved state, independent of late cancel and
 // loader validity. Actual issue/ready still enforce every original guard.
 wire correction_operand=need_correction && !preparing;
 wire start_correction=selected_eligible && correction_operand;''')
    text=once(text,' wire [31:0] split_lane;\n // All four spatial rows', ' // All four spatial rows')
    text=once(text,' wire issue=(|eligible) && (!need_correction || (preparing && correction_wait==0));',
        ' assign issue=selected_eligible && (!need_correction || (preparing && correction_wait==0));')
    text=once(text,'wire post_grant=!preparing && eligible[2];','wire post_grant=selected_eligible && chosen==2;')
    text=once(text,'wire align_grant=!preparing && !eligible[2] && eligible[1];','wire align_grant=selected_eligible && chosen==1;')
    text=once(text,'wire [17:0] selected_scale=start_correction ?', 'wire [17:0] selected_scale=correction_operand ?')
    text=once(text,"wire [18:0] operand_a=start_correction ?", "wire [18:0] operand_a=correction_operand ?")
    text=once(text,' assign busy=preparing || (|nonempty)||(|token_valid);',' assign busy=selected || preparing || (|nonempty)||(|token_valid);')
    text=once(text,' || (preparing && !in_valid[0]))error<=1;', ' || (selected && !in_valid[chosen]))error<=1;')
    return text

def audit_delta(board):
    audit_parent(BASE)
    meta=json.loads((board/'inputs.json').read_text());changed=[]
    for rec in meta['inputs']:
        f=Path(rec['snapshot']);assert sha(f)==rec['sha256'];rel=f.relative_to(board)
        if sha(f)!=sha(BASE/rel):changed.append(rel.as_posix())
    assert sorted(changed)==sorted('assembly/'+n for n in CHANGED),changed
    for name,fn in zip(CHANGED,(router,rq)):
        assert (board/'assembly'/name).read_text()==fn((BASE/'assembly'/name).read_text())
    return dict(changed=changed,full_camera_ddr_cnn_display_preserved=True,
        clock_mhz=150,pins_clocks_sdc_phy_unchanged=True,added_arbitration_cycles=True,
        hardware_access=False,board_release=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--map-prepared',action='store_true');a=p.parse_args();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp')
    if not a.map_prepared:
        assert not root.exists();audit_parent(BASE);root.mkdir()
        for d in ('rtl','assembly'):shutil.copytree(BASE/d,root/d)
        for f in BASE.iterdir():
            if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
        for n,fn in zip(CHANGED,(router,rq)):(root/'assembly'/n).write_text(fn((BASE/'assembly'/n).read_text()),encoding='utf-8')
        meta=json.loads((BASE/'inputs.json').read_text())
        for rec in meta['inputs']:
            f=root/Path(rec['snapshot']).relative_to(BASE);rec.update(snapshot=str(f),sha256=sha(f))
        meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),ownership_runner_sha256=sha(__file__),board_download_allowed=False)
        (root/'inputs.json').write_text(json.dumps(meta,indent=2));shutil.copy2(__file__,root/'ownership_runner.py')
        (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2));print('OWNERSHIP_BOARD_PREPARED_NOT_RELEASE');return
    audit_delta(root);assert not (root/'map.log').exists()
    (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow map --timeout 900\nexit /b %errorlevel%\n')
    start=time.monotonic()
    try:
        with (root/'map.log').open('x') as log:
            proc=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            (root/'progress.json').write_text(json.dumps(dict(stage='map',pid=proc.pid,budget_s=960)))
            try:code=proc.wait(timeout=960)
            except subprocess.TimeoutExpired:stop_launched_tree(proc);raise
        assert code==0 and re.search(r'map\s*:\s*PASS',(root/'map.log').read_text())
        net=root/'outflow/ti60f225_oob.map.v'
        state=audit_state(net,'cpu/','process_clk_150')
        sync=audit_sync(net.read_text(),prefix='cpu/system_hardJtag_debug_logic_jtagBridge/observation/',system='process_clk_150',tck='jtag_inst1_TCK')
        (root/'control_state_map_audit.json').write_text(json.dumps(state,indent=2))
        (root/'snapshot_sync_audit.json').write_text(json.dumps(sync,indent=2))
        warnings=(root/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
        assert 'EFX-0654' not in warnings
        audit_delta(root)
        result=dict(status='REGISTERED_OWNERSHIP_FULL_BOARD_MAP_NOT_RELEASE',seconds=time.monotonic()-start,
            inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),snapshot_sync_audit_sha256=sha(root/'snapshot_sync_audit.json'),
            physical_pass=False,board_download_allowed=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
