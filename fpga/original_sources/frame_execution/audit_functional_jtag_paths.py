"""Inventory the functional JTAG CDC separately from the observation channel.

Command data is held while its toggle crosses two system FFs. Replies are
held until the next command, and need the bounded host's stable two-scan read.
This is a digital structural/settling check, not an analogue MTBF guarantee.
"""
import argparse,json,re
from pathlib import Path
from run_units import sha
from audit_command_toggle_sync import audit as sync_audit
from audit_snapshot_jtag_paths import PREFIX as OBSERVATION

def audit(root):
    net=root/'outflow/ti60f225_oob.map.v'
    sync=sync_audit(net.read_text());assert not sync['errors'],sync
    first=next(n for n,c in sync['cells'].items() if c['ports']['D']==sync['source'])+'|D'
    payload={'cpu/remote_bit~FF|D','cpu/remote_last~FF|D'}
    response_prefix='cpu/system_hardJtag_debug_logic_jtagBridge/jtag_readArea_full_shifter'
    replies={response_prefix+f'[{i}]~FF|D' for i in range(34)}
    rows=[];reports={}
    for direction in ('jtag_to_system','system_to_jtag'):
        path=root/f'outflow/ti60f225_oob.{direction}.rpt'
        body=path.read_text();reports[str(path)]=sha(path)
        assert '-npaths 10000 -nworst 1' in body
        blocks=body.split('++++ Path ')[1:];assert len(blocks)<10000
        for b in blocks:
            def field(n):
                m=re.search(r'^'+re.escape(n)+r'\s*:\s*(.+)$',b,re.M);assert m,n
                return m[1].strip()
            start,end=field('Path Start'),field('Path End')
            if end.startswith(OBSERVATION):continue
            delay=float(field('Data Path Delay').split()[0])
            if direction=='jtag_to_system' and end==first:
                kind='command_toggle_first_sync_ff'
            elif direction=='jtag_to_system' and end in payload:
                kind='command_bundled_data_after_two_system_stages'
                assert 'inputArea_data_' in start and start.endswith('~FF|CLK'),(start,end)
                # Earliest payload sampling follows two system clock edges.
                # Reserve more than one full 150 MHz cycle for other effects.
                assert delay<6.0,(start,end,delay)
            elif direction=='system_to_jtag' and end in replies:
                kind='held_reply_requires_two_matching_host_scans'
                assert 'system_rsp_' in start and start.endswith('~FF|CLK'),(start,end)
                assert delay<100.0,(start,end,delay)
            else:raise AssertionError(('Unclassified functional JTAG crossing',direction,start,end))
            rows.append(dict(start=start,end=end,delay_ns=delay,kind=kind))
    assert {r['end'] for r in rows}==payload|replies|{first}
    return dict(status='FUNCTIONAL_JTAG_ROUTED_SETTLING_PASS_WITH_STABLE_HOST_CONTRACT',
                map_sha256=sha(net),synchronizer=sync,paths=rows,reports=reports,
                command_payload_max_ns=max(r['delay_ns'] for r in rows if r['end'] in payload),
                response_max_ns=max(r['delay_ns'] for r in rows if r['end'] in replies),
                host_contract=dict(tck_max_hz=1000000,consecutive_matching_valid_scans=2,
                    no_new_command_until_response=True,reply_scan_bits=36),
                unknown_functional_crossings=0,analogue_metastability_proven=False,board_release=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    result=audit(a.root.resolve());out=a.root/'functional_jtag_cdc.json'
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(result['status'])
