"""Bind every non-register synchronizer-head warning to its source class.

This inventory is not a CDC waiver. In particular, an async_reg attribute on
the legacy FIFO reset counters is not evidence of a conventional two-FF chain.
"""
import argparse, collections, json, re
from pathlib import Path
from run_units import ROOT, sha

p=argparse.ArgumentParser()
p.add_argument('--board',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.board.resolve();out=a.output.resolve()
assert root.is_relative_to(ROOT/'tmp') and out.is_relative_to(ROOT) and not out.exists()
meta=json.loads((root/'physical_inputs.json').read_text())
net=root/'outflow/ti60f225_oob.map.v'
assert sha(net)==meta['base_map_sha256']
log=root/'outflow/ti60f225_oob.place.out'
pattern=r'Synchronizer chain starting at cell (\S+) is driven by a non-sequential source (\S+) \[FlowMsg::SyncHeadDrivenByNonSeq\]'
warnings=re.findall(pattern,log.read_text())
assert len(warnings)==len(set(warnings))==23,('Unexpected warning inventory',len(warnings))
rows=[]
for cell,source in warnings:
    if source=='GND:O':
        assert cell.startswith(('cpu/','inst_efx_csi2_rx/'))
        kind='constant_input_no_toggling_data'
    elif re.search(r'/fifo/(Wr|Rd)ClkRstGen\[[01]\]~FF$',cell):
        kind='legacy_fifo_reset_handshake_counter'
    elif cell=='raw_frontend/ingress/ack_meta~FF':
        kind='held_abort_acknowledgement'
    elif cell=='raw_frontend/abort_sync/request_meta~FF':
        kind='held_abort_request'
    elif cell=='u_ddr3_top/ready_meta~FF':
        kind='combined_vendor_calibration_status'
    else:
        raise AssertionError(('Unclassified synchronization warning',cell,source))
    rows.append(dict(cell=cell,driver=source,source_class=kind))
counts=dict(collections.Counter(r['source_class'] for r in rows))
assert counts==dict(constant_input_no_toggling_data=8,legacy_fifo_reset_handshake_counter=12,
                   held_abort_acknowledgement=1,held_abort_request=1,combined_vendor_calibration_status=1)
sources=[root/'assembly/raw_stream_ingress.sv',root/'assembly/guarded_async_fifo.sv',
         root/'rtl/frame_buffer_V3/DC_FIFO.v',root/'assembly/EfxSapphireSoc.v',root/'ddr516_v4_controller.sv']
assert all(f.exists() for f in sources)
data=dict(status='ALL_SYNC_HEAD_WARNINGS_CLASSIFIED_NOT_CDC_RELEASE',
          map_sha256=sha(net),physical_inputs_sha256=sha(root/'physical_inputs.json'),
          log_sha256=sha(log),counts=counts,heads=rows,
          source_sha256={str(f):sha(f) for f in sources},
          remaining_checks=[
              'Reset assertion/release coverage including both FIFO clock domains and the 20-cycle guards.',
              'Abort request/ack stability until four-phase cancellation drains outstanding traffic.',
              'Vendor calibration_done/calibration_pass transition contract at the ready synchronizer.',
              'Actual routed Gray skew and same-domain synchronizer timing; never accept empty skew reports.'
          ],exceptions_added=False,full_cdc_pass=False,board_release=False)
out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=data['status'],counts=counts)))
