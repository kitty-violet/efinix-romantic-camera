"""Add only first-failure observation to the already routed 150 MHz board."""
import hashlib,json
from pathlib import Path
from run_units import ROOT,HERE,sha
BASE=ROOT/'tmp/frame-execution/connected-board-registered-ownership2'
MODULE=HERE/'variants/board_first_fault/control_first_fault.sv'
OLD_ASSIGN=''' assign observation_payload={diagnostic_board_status,observation_flags,
  observation_reply,observation_address,
  observation_replies,observation_accepts,observation_last,observation_bits,observation_ticks};'''
NEW_ASSIGN=''' // First-fault view: functional controller, requests and reset are unchanged.
 control_first_fault first_fault(.clk(clk),.reset(reset),
  .control_state(state_oh),.boot_state(boot_observer_state),.address(address),
  .flags(observation_flags),.bus_age(bus_age),.fatal(fatal_bus_error),
  .boot_failed(boot_failed),.record_data(observation_payload));'''
def transform_soc(text):
    assert text.count(OLD_ASSIGN)==1
    for old,new in (
      (' output wire [31:0] observed_crc\n);',' output wire [31:0] observed_crc,\n output wire [5:0] observer_state\n);'),
      (' assign observed_crc=~crc;',' assign observed_crc=~crc;\n assign observer_state=state;'),
      (' wire [7:0] boot_fault;wire boot_terminal,boot_failed;',' wire [7:0] boot_fault;wire boot_terminal,boot_failed;\n wire [5:0] boot_observer_state;'),
      ('.fault(boot_fault),.observed_crc());','.fault(boot_fault),.observed_crc(),.observer_state(boot_observer_state));')):
        assert text.count(old)==1,old
        text=text.replace(old,new)
    return text.replace(OLD_ASSIGN,NEW_ASSIGN)+'\n'+MODULE.read_text(encoding='utf-8')
def transform_observer(text):
    assert text.count("32'h4f425331")==1
    return text.replace("32'h4f425331","32'h4f465331")
def audit_delta(board):
    board=Path(board)
    meta=json.loads((board/'inputs.json').read_text())
    changed=[]
    for rec in meta['inputs']:
        p=Path(rec['snapshot']);assert p.is_relative_to(board) and sha(p)==rec['sha256']
        rel=p.relative_to(board)
        if sha(p)!=sha(BASE/rel):changed.append(rel.as_posix())
    assert sorted(changed)==['assembly/EfxSapphireSoc.v','assembly/jtag_bringup_snapshot.sv'],changed
    assert (board/'assembly/EfxSapphireSoc.v').read_text()==transform_soc((BASE/'assembly/EfxSapphireSoc.v').read_text())
    assert (board/'assembly/jtag_bringup_snapshot.sv').read_text()==transform_observer((BASE/'assembly/jtag_bringup_snapshot.sv').read_text())
    for rel in ('ti60f225_oob.xml','ti60f225_oob.sdc','ti60f225_oob.peri.xml','rtl/ti60f225_oob_top.v'):
        assert sha(board/rel)==sha(BASE/rel),rel
    return dict(status='FIRST_FAULT_OBSERVATION_ONLY_DELTA',changed=changed,
                functional_rtl_unchanged=True,pins_clocks_sdc_unchanged=True,
                clock_mhz=150,full_video_a_chain_preserved=True,board_release=False)
