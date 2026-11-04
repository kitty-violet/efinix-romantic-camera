"""Bind the video-only leaf specialization to its four-state output test and map."""
import argparse,json
from pathlib import Path
from run_units import ROOT,sha
from audit_connected_system import manifest
from audit_resource_scope import read_board

AUDIO={'audio_packet.sv','audio_infoframe_packet.sv','audio_clock_regeneration_packet.sv'}

def verify_no_audio_binding(binding):
    assert binding['status']=='NO_AUDIO_SOURCE_BINDING_PASS'
    board=Path(binding['board']);reference=Path(binding['reference_board'])
    before=read_board(reference);after=read_board(board)
    assert before['names'].keys()==after['names'].keys()
    changed={n for n in before['names'] if before['names'][n]!=after['names'][n]}
    assert changed==AUDIO and len(after['names'])==125
    assert not after['result']['new_undriven_signatures']
    assert binding['inputs_sha256']==after['result']['inputs_sha256']
    assert binding['map_sha256']==after['result']['map_sha256']
    prior=binding['inherited_binding']
    assert sha(prior['path'])==prior['sha256']
    inherited=json.loads(Path(prior['path']).read_text())
    assert inherited['status']=='CONTROL_PAYLOAD_SOURCE_CYCLE_BINDING_PASS'
    assert inherited['inputs_sha256']==before['result']['inputs_sha256']
    assert inherited['map_sha256']==before['result']['map_sha256']
    for record in inherited['gates']:
        assert sha(Path(record['path'])/'result.json')==record['result_sha256']
    assert len(binding['gates'])==1
    record=binding['gates'][0];root=Path(record['path'])
    assert sha(root/'result.json')==record['result_sha256']
    result=json.loads((root/'result.json').read_text())
    assert result['status']=='NO_AUDIO_HDMI_FOUR_STATE_AB_PASS'
    assert sha(root/'inputs.json')==result['inputs_sha256']==record['inputs_sha256']
    tested=json.loads((root/'inputs.json').read_text())
    assert tested['base_inputs_sha256']==before['result']['inputs_sha256']
    for side,expected in [('reference',before['names']),('candidate',after['names'])]:
        rows=[r for r in tested['inputs'] if r['variant']==side]
        names={Path(r['snapshot']).name:r['sha256'] for r in rows}
        for row in rows:assert sha(row['snapshot'])==row['sha256']
        common=names.keys()&expected.keys()
        assert AUDIO<=common and {'dvi_encoder.v_encrypted.v','tx_aux.v_encrypted.v',
                                 'auxiliary_video_information_info_frame.sv','packet_assembler.sv'}<=common
        mismatch=[n for n in common if names[n]!=expected[n]]
        assert not mismatch, (side,mismatch)
    for capture in tested['captures']:assert sha(capture['path'])==capture['sha256']
    assert result['avi_identical'] and result['non_island_symbols_identical'] and result['all_packet_bch_valid']
    return before,after

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True)
    p.add_argument('--reference-board',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True);a=p.parse_args()
    board=a.board.resolve();ref=a.reference_board.resolve();gate=a.gate.resolve()
    output=board/'gate_binding.json';assert board.is_relative_to(ROOT/'tmp') and not output.exists()
    br=json.loads((board/'result.json').read_text());gr=json.loads((gate/'result.json').read_text())
    binding=dict(status='NO_AUDIO_SOURCE_BINDING_PASS',board=str(board),reference_board=str(ref),
      inputs_sha256=br['inputs_sha256'],map_sha256=br['map_sha256'],source_names=125,
      changed_sources=sorted(AUDIO),unchanged_sources=122,
      inherited_binding=dict(path=str(ref/'gate_binding.json'),sha256=sha(ref/'gate_binding.json')),
      gates=[dict(path=str(gate),result_sha256=sha(gate/'result.json'),inputs_sha256=gr['inputs_sha256'])],
      full_frame=False,physical_timing_pass=False,resource_fit=False,board_download_allowed=False)
    before,after=verify_no_audio_binding(binding)
    binding['resource_delta']={k:v-before['result']['resources'].get(k,0) for k,v in after['result']['resources'].items()}
    output.write_text(json.dumps(binding,indent=2)+'\n')
    print(json.dumps(binding),flush=True)

if __name__=='__main__':main()
