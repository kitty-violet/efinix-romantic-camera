"""Bind completed replacement serial proof without rewriting failed attempts."""
import json,shutil
from pathlib import Path
from run_units import ROOT,sha
from finish_fault_probe_release import SHORTS,SERIAL,ADAPTER
def main():
    board=ROOT/'tmp/frame-execution/connected-board-fault-probe1';mapped=sha(board/'outflow/ti60f225_oob.map.v')
    evidence={}
    for folder,status in ((SHORTS/'startup','COMMAND_PROGRESS_MAPPED_STARTUP_PASS_PENDING_SERIAL_FAULT'),
        (SHORTS/'observe','MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS'),
        (SERIAL,'MAPPED_SERIAL_APB_OFS_RESET_PASS'),
        (ADAPTER,'ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS')):
        r=json.loads((folder/'result.json').read_text());i=json.loads((folder/'inputs.json').read_text())
        assert r['status']==status and r['inputs_sha256']==sha(folder/'inputs.json')
        assert i.get('source_map_sha256',i.get('original_map_sha256'))==mapped
        assert r.get('log_sha256',r.get('simulation_sha256'))==sha(folder/'simulation.log')
        for n,h in i['files'].items():assert sha(folder/n)==h
        evidence[str(folder/'result.json')]=sha(folder/'result.json')
    for n in ('progress_counter_checks.json','paired_reply_result.json'):
        assert not (SERIAL/n).exists();shutil.copy2(SERIAL.parent/n,SERIAL/n)
        evidence[str(SERIAL/n)]=sha(SERIAL/n)
    i=json.loads((SERIAL/'inputs.json').read_text());assert i['tck_hz']==10000000
    p=SHORTS/'result.json';assert not p.exists()
    p.write_text(json.dumps(dict(status='FAULT_PROBE_SAME_MAP_STARTUP_OBSERVE_SERIAL_PASS',
        map_sha256=mapped,completed_evidence=evidence,runner_sha256=sha(__file__),
        prior_failure_preserved=True,recovery='Retained passing startup+observer. Full serial at original10MHz TCK constraint; official public models textually inlined. No hardware/PHY timing claim.',
        hardware_access=False),indent=2))
    print('FAULT_PROBE_SAME_MAP_STARTUP_OBSERVE_SERIAL_PASS')
if __name__=='__main__':main()
