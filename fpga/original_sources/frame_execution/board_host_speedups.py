"""Admit measured four-state host optimizations without modifying hardware RTL."""
import json
import shutil
from pathlib import Path
from run_units import sha
from board_ring_scoreboard import transform


def proof(root, status):
    root=root.resolve()
    result=json.loads((root/'result.json').read_text())
    inputs=json.loads((root/'inputs.json').read_text())
    assert result['status']==status
    assert result['inputs_sha256']==sha(root/'inputs.json')
    assert Path(inputs['simulator_bin']).resolve()==Path('E:/quatra/questa_fse/win64').resolve(), 'Require the actual Questa simulator proof'
    for name,row in result['measurements'].items():
        assert sha(root/(name+'.log'))==row['log_sha256']
    return root,result,inputs


def retain_proof(board,manifest,source,name,details):
    target=board/name
    shutil.copy2(source/'result.json',target)
    manifest['inputs'].append(dict(source=str(source/'result.json'),snapshot=str(target),sha256=sha(target)))
    manifest.setdefault('host_speedups',{})[name]=dict(proof_root=str(source),
        proof_inputs_sha256=sha(source/'inputs.json'),hardware_changed=False,**details)


def ram_word(board,manifest,proof_root,simple=False):
    source,result,inputs=proof(proof_root,'PUBLIC_RAM10_WORD_EQUIVALENCE_PASS' if simple else 'PUBLIC_DPRAM_WORD_EQUIVALENCE_PASS')
    target=board/('efx_ram10.v' if simple else 'efx_dpram10.v')
    assert sha(target)==inputs['official_sha256']
    assert sha(source/'word.v')==inputs['word_sha256']
    assert sha(source/'tb.sv')==inputs['tb_sha256']
    shutil.copy2(source/'word.v',target)
    row=next(row for row in manifest['inputs'] if Path(row['snapshot'])==target)
    row.update(source=str(source/'word.v'),sha256=sha(target),
               vendor_original_sha256=inputs['official_sha256'],
               view='Four-state checked public equal-width RAM word storage; original path for other widths')
    retain_proof(board,manifest,source,'simple_ram_word_proof.json' if simple else 'ram_word_proof.json',
                 dict(isolated_host_speed_ratio=result['host_speed_ratio'],whole_board_speed_measured=False))


def ring_checker(board,manifest,proof_root,tb):
    source,result,inputs=proof(proof_root,'BOARD_TMDS_RING_CHECKER_EQUIVALENCE_PASS')
    assert inputs['helper_sha256']==sha(Path(__file__).with_name('board_ring_scoreboard.py'))
    for name,key in [('old.sv','old_sha256'),('ring.sv','ring_sha256'),('tb.sv','test_sha256')]:
        assert sha(source/name)==inputs[key]
    assert result['latency_clock_edges']==58
    assert result['measurements']['shift_speed']['checksum']==result['measurements']['ring_speed']['checksum']
    changed=transform(tb)
    retain_proof(board,manifest,source,'ring_checker_proof.json',
                 dict(isolated_host_speed_ratio=result['isolated_host_speed_ratio'],whole_board_speed_measured=False))
    return changed
