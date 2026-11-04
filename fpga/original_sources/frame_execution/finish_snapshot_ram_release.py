"""Full offline release gates for the single RAM snapshot storage change.

No programming is performed. Numeric datapath evidence is inherited only
through the exact one-file delta; RAM snapshot/CDC and all mapped control
checks are current. Failed and pre-change records remain immutable.
"""
import inspect,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from snapshot_ram_board import BOARD,BASE as REGISTERED_PARENT,BRIDGE,proofs,audit_delta
from finish_split_logic_outputs_release import archived_placement_evidence
import finish_registered_boundary_release as prior
BASE=prior.BASE
PARENT=prior.PARENT
DATAPATH_PARENT=prior.DATAPATH_PARENT
SHORTS=ROOT/'tmp/frame-execution/snapshot-ram-evidence2'
RTL=BRIDGE
SERIAL=ROOT/'tmp/frame-execution/snapshot-ram-evidence2-serial/new'
MAP_AUDIT=SHORTS/'map-audit'
ADAPTER=ROOT/'tmp/frame-execution/snapshot-ram-evidence2-adapter'
STATUS=prior.STATUS

def parent_evidence(parent,verify):
 assert Path(parent['board'])==BASE;verify(parent['required_evidence']);audit_delta(BOARD)
 return dict(status='RELEASE_PARENT_AND_RAM_SNAPSHOT_DELTA_BOUND',manifest=str(PARENT/'manifest.json'),
  sha256=sha(PARENT/'manifest.json'),scope='Full video/A preserved via audited parents; new snapshot and exact map gates required; no inherited board PASS.')

def verify_wrapper(path,source,wrapper,bind,verify):
 meta=bind(path.with_suffix('.json'))
 assert Path(meta['source'])==source
 verify({str(path):meta['driver_sha256'],str(source):meta['source_sha256'],str(wrapper):meta['wrapper_sha256']})

def extra_evidence(board,batch,physical,serial,bind,verify):
 # Keep all established control, adapter, finite primitive, inherited
 # datapath/A and physical checks. Rebind the two generated wrapper layers.
 code=inspect.getsource(prior.extra_evidence)
 code=code.replace("'REGISTERED_BOUNDARY_SAME_MAP_SHORT_INTEGRATION_PASS'","'SNAPSHOT_RAM_SAME_MAP_SHORT_INTEGRATION_PASS'")
 code=once(code,"ai['board_inputs_sha256']==sha(board/'inputs.json')","ai['board_inputs_sha256']==sha(REGISTERED_PARENT/'inputs.json')")
 code=once(code,"str(HERE/'registered_boundary_route.py'):dm['wrapper_sha256']",
  "str(batch.parent/(batch.name+'-ram-wrapper.py')):dm['wrapper_sha256']")
 code=once(code,"str(HERE/'place_registered_boundary_board.py'):wm['wrapper_sha256']",
  "str(physical.parent/(physical.name+'-ram-wrapper.py')):wm['wrapper_sha256']")
 scope=dict(prior.__dict__)
 scope.update(BOARD=BOARD,BASE=BASE,DATAPATH_PARENT=DATAPATH_PARENT,REGISTERED_PARENT=REGISTERED_PARENT,
  SHORTS=SHORTS,RTL=RTL,SERIAL=SERIAL,MAP_AUDIT=MAP_AUDIT,ADAPTER=ADAPTER,
  audit_delta=audit_delta,__file__=__file__)
 exec(compile(code,__file__+':inherited-gates','exec'),scope)
 scope['extra_evidence'](board,batch,physical,serial,bind,verify)
 verify_wrapper(batch.parent/(batch.name+'-ram-wrapper.py'),HERE/'registered_boundary_route.py',
  HERE/'route_snapshot_ram_board.py',bind,verify)
 verify_wrapper(physical.parent/(physical.name+'-ram-wrapper.py'),HERE/'place_registered_boundary_board.py',
  HERE/'place_snapshot_ram_board.py',bind,verify)
 # Prior serial.log belongs to the old wide register snapshot; it is history,
 # not the new RAM proof. The other fourteen cases concern unchanged RTL.
 old=ROOT/'tmp/frame-execution/split-outputs-shorts2'
 for name in ('bringup_cell_probe.sv','bringup_clock_probe.sv','sapphire_axi32_to_axi128.sv'):
  assert sha(old/name)==sha(board/'assembly'/name)
 for path,h in proofs().items():verify({path:h})
 from snapshot_ram_board import SHORT
 for folder in (SHORT,BRIDGE):
  r=bind(folder/'result.json');m=bind(folder/'inputs.json')
  verify({row['snapshot']:row['sha256'] for row in m['inputs']})
  assert r['inputs_sha256']==sha(folder/'inputs.json')
  assert sha(folder/'jtag_bringup_snapshot.sv')==sha(board/'assembly/jtag_bringup_snapshot.sv')
  if folder==SHORT:
   verify({str(folder/f"seed{x['seed']}.log"):x['log_sha256'] for x in r['results']})
  else:
   assert sha(folder/'EfxSapphireSoc.v')==sha(board/'assembly/EfxSapphireSoc.v')
   verify({str(folder/'simulation.log'):r['log_sha256']})
 from snapshot_ram_cdc import structure
 assert structure(board)==bind(MAP_AUDIT/'snapshot_ram_structure.json')
 for name in ('snapshot_ram_cdc.py','snapshot_ram_board.py','prepare_snapshot_ram_board.py',
  'prepare_board_variant.py','run_snapshot_ram_shorts.py','run_snapshot_ram_bridge.py',
  'run_snapshot_ram_evidence.py','snapshot_ram_flow.py','audit_snapshot_ram_map.py',
  'audit_snapshot_ram_gray.py','complete_snapshot_ram_map_audit.py','finish_snapshot_ram_release.py'):
  verify({str(HERE/name):sha(HERE/name)})

def package_code():
 source,code=prior.package_code()
 code=once(code,'from registered_boundary_board import audit_delta','from snapshot_ram_board import audit_delta')
 code=once(code,'from finish_registered_boundary_release import BASE,SHORTS,RTL,SERIAL,extra_evidence,parent_evidence',
  'from finish_snapshot_ram_release import BASE,SHORTS,RTL,SERIAL,extra_evidence,parent_evidence,REGISTERED_PARENT')
 code=once(code,'from audit_snapshot_jtag_paths import audit as snapshot_cdc','from snapshot_ram_cdc import audit as snapshot_cdc')
 code=once(code,"verify({str(HERE/'prepare_split_registered_board.py'):mi['registered_boundary_transform_sha256']})",
  "assert mi['base_inputs_sha256']==sha(REGISTERED_PARENT/'inputs.json')\n    verify({str(HERE/'prepare_snapshot_ram_board.py'):sha(HERE/'prepare_snapshot_ram_board.py'),str(HERE/'prepare_board_variant.py'):sha(HERE/'prepare_board_variant.py')})")
 code=once(code,"'CONTROL_DDR_RTL_SERIAL_EVENTS_RESET_PASS'","'OFP5_RAM_REAL_BRIDGE_RESET_PROTOCOL_PASS'")
 before="        verify({str(folder/n):h for n,h in inputs['files'].items()})"
 after="""        if 'files' in inputs:verify({str(folder/n):h for n,h in inputs['files'].items()})
        else:
            assert folder==RTL
            verify({r['snapshot']:r['sha256'] for r in inputs['inputs']})"""
 code=once(code,before,after)
 code=once(code,"if 'board_inputs_sha256' in inputs:assert inputs['board_inputs_sha256']==sha(board/'inputs.json')",
  "if 'board_inputs_sha256' in inputs:assert inputs['board_inputs_sha256']==sha((REGISTERED_PARENT if folder==RTL else board)/'inputs.json')")
 code=once(code,"str(HERE/'finish_registered_boundary_release.py'):driver_meta['adapter_sha256']",
  "str(HERE/'finish_snapshot_ram_release.py'):driver_meta['adapter_sha256']")
 code=code.replace('零COMB4与RQ/source寄存边界的完整视频/A候选','零COMB4、寄存边界与RAM快照的完整视频/A候选')
 code=once(code,'independent_outputs_workaround=True,','independent_outputs_workaround=True,snapshot_storage="dual_bank_native_RAM",')
 return source,code

def finish_code():
 source,code=prior.finish_code()
 code=code.replace("'finish_registered_boundary_release.py'","'finish_snapshot_ram_release.py'")
 return source,code

def main():
 mode=sys.argv.pop(1)
 if mode=='pll':
  import audit_pll_shift_routed as check
  check.audit_delta=audit_delta;check.main();return
 if mode=='bundle':source,code=package_code();key='--output'
 else:
  assert mode=='finish';source,code=finish_code();key='--root'
 args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve()
 delta=audit_delta(BOARD);record=BOARD/'delta_audit.json'
 if record.exists():assert json.loads(record.read_text())==delta
 else:record.write_text(json.dumps(delta,indent=2))
 driver=out.parent/(out.name+'-ram-driver.py');assert not driver.exists()
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
  generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
 sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
