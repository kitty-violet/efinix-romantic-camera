"""Bind measured complete CNN and the same implemented whole-board resource set."""
import argparse,json,re,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,sha
from audit_connected_system import manifest
parser=argparse.ArgumentParser()
parser.add_argument('--board',type=Path,required=True);parser.add_argument('--frame',type=Path,required=True)
parser.add_argument('--physical',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();b=args.board.resolve();f=args.frame.resolve();p=args.physical.resolve()
assert all(x.is_relative_to(ROOT/'tmp') for x in (b,f,p))
bm=json.loads((b/'inputs.json').read_text());fm,fs=manifest(f)
# A frozen board also retains vendor originals alongside selected wrappers.
# Verify all files, then bind RTL names only from the actual build source list.
for row in bm['inputs']:assert sha(row['snapshot'])==row['sha256']
xml=ET.parse(b/'ti60f225_oob.xml');ns={'e':'http://www.efinixinc.com/enf_proj'}
selected={(b/n.attrib['name']).resolve() for n in xml.findall('.//e:design_file',ns)}
assert selected=={Path(n).resolve() for n in bm['source_files']}
bs={}
for row in bm['inputs']:
    source=Path(row['snapshot']).resolve()
    if source not in selected:continue
    assert source.name not in bs or bs[source.name]==row['sha256'],source.name
    bs[source.name]=row['sha256']
br=json.loads((b/'result.json').read_text());fr=json.loads((f/'result.json').read_text())
assert sha(b/'inputs.json')==br['inputs_sha256'] and sha(f/'inputs.json')==fr['inputs_sha256']
assert sha(b/'outflow/ti60f225_oob.map.v')==br['map_sha256']
assert sha(b/'outflow/ti60f225_oob.map.rpt')==br['report_sha256']
assert sha(f/'simulation.log')==fr['log_sha256'] and sha(fm['binary'])==fm['binary_sha256']
bench={'efx_dpram10.v','efx_dsp24.v','efx_dsp48.v','efx_ram10.v','efx_srl8.v','efx_lut4.v','tb_fourway_full.sv'}
common=fs.keys()-bench
assert len(common)==44 and common<=bs.keys()
for n in common:assert fs[n]==bs[n],n
assert fr['status']=='CONTINUOUS_720P_RTL_NUMERIC_MEASUREMENT_PASS'
assert fr['cycles']>0 and fr['commands']==19229
line=fr['line'];assert int(line[3])==65952000 and int(line[4])==2764800 and line[-1]=='1fffffff'
pm=json.loads((p/'physical_inputs.json').read_text());assert pm['base_map_sha256']==br['map_sha256']
for r in pm['inputs']:assert sha(r['snapshot'])==r['sha256']
dsp_layout=pm.get('dsp_layout')
gray_layout=pm.get('related_gray_layout')
gray_audit_sha=None
if dsp_layout:
    assert sha(p/'fixed_dsp.placeloc')==dsp_layout['assignments_sha256']
    assert sha(p/'efx_pnr_settings.ini')==dsp_layout['settings_sha256']
    assert sha(dsp_layout['reference_netlist'])==dsp_layout['reference_netlist_sha256']
    assert sha(dsp_layout['reference_placement'])==dsp_layout['reference_placement_sha256']
    assignments=(p/'fixed_dsp.placeloc').read_text().splitlines()
    assert len(assignments)==len(set(assignments))
    if gray_layout:
        proof=p/'related_gray_placement.json'
        assert json.loads(proof.read_text())==gray_layout
        assert gray_layout['sources_only'] and len(gray_layout['pairs'])==7
        assert gray_layout['reference_map_sha256']==br['map_sha256']
        assert gray_layout['assignments_sha256']==dsp_layout['assignments_sha256']
        audit_file=p/'related_gray_placement_audit.json'
        audit=json.loads(audit_file.read_text())
        gray_audit_sha=sha(audit_file)
        assert audit['status']=='PUBLIC_SOURCE_ASSIGNMENTS_VERIFIED_NOT_TIMING_PASS'
        assert audit['sources']==7 and audit['protected_heads_are_automatic']
        assert audit['proof_sha256']==sha(proof)
        assert audit['placement_sha256']==sha(p/'outflow/ti60f225_oob.out.placeloc')
        assert {r['bit'] for r in gray_layout['pairs']}==set(range(7))
        assert len(audit['pairs'])==7 and {r['bit'] for r in audit['pairs']}==set(range(7))
        actual={r['bit']:r for r in audit['pairs']}
        extras=[]
        for row in gray_layout['pairs']:
            assert actual[row['bit']]['source']==row['source_location']
            extras.append(f"{row['source']} {' '.join(map(str,row['source_location']))}")
        assert len(set(extras))==7 and set(extras)<=set(assignments)
        assignments=[line for line in assignments if line not in set(extras)]
    # Extra public FF assignments must be proved above; all remaining entries
    # must be DSP primitives from this exact mapped design, not arbitrary cells.
    dsp_names=set(re.findall(r'^\s*EFX_DSP(?:24|48)\s+\\(\S+)\s',
                            (p/'outflow/ti60f225_oob.map.v').read_text(),re.M))
    assigned_names=[line.split()[0] for line in assignments]
    assert len(assigned_names)==dsp_layout['dsp24']+dsp_layout['dsp48']
    assert len(set(assigned_names))==len(assigned_names) and set(assigned_names)==dsp_names
else:
    assert not gray_layout,'Public FF placement requires the validated DSP layout'
log=p/'outflow/ti60f225_oob.place.out';body=log.read_text()
m=re.search(r'XLR count after merging:\s*(\d+)',body);assert m
packed=int(m.group(1));capacity=60800
report=p/'outflow/ti60f225_oob.place.rpt';physical=report.read_text()
m=re.search(r'^XLRs:\s*(\d+)\s*/\s*(\d+)',physical,re.M);assert m
used,physical_capacity=map(int,m.groups());assert physical_capacity==capacity
m=re.search(r'XLRs needed for Routing:\s*(\d+)',physical);assert m
routing=int(m.group(1));assert used>=packed and used>=routing
data=dict(status='FIXED_PRODUCTION_MEASUREMENT_BOUND_NOT_TARGET_PASS',frame=str(f),board=str(b),physical=str(p),
    inputs_sha256=fr['inputs_sha256'],log_sha256=fr['log_sha256'],frame_result_sha256=sha(f/'result.json'),
    board_inputs_sha256=br['inputs_sha256'],map_sha256=br['map_sha256'],packing_log_sha256=sha(log),
    shared_rtl={n:fs[n] for n in sorted(common)},frame_cycles=fr['cycles'],ms_at_assumed_150MHz=fr['measured_frame_ms'],
    fps_at_assumed_150MHz=150000000/fr['cycles'],resources=br['resources'],packed_logic=packed,logic_capacity=capacity,
    actual_xlr_including_routing=used,routing_xlr=routing,physical_report_sha256=sha(report),
    ram=br['resources'].get('RAM10',0)+br['resources'].get('DPRAM10',0),
    physical_dsp=br['resources'].get('DSP48',0)+(br['resources'].get('DSP24',0)+1)//2,
    cycle_deficit_30fps=fr['cycles']-5000000,logic_excess=used-capacity,
    scope=fr['scheduling'],ddr=fr['ddr'],weights=fr['weights'],full_baseline=False,full_csi_to_tmds=False,
    gesture_demon_real_fps='NOT_INTEGRATED_BUDGET_OPEN',dsp_layout=dsp_layout,related_gray_layout=gray_layout,
    related_gray_audit_sha256=gray_audit_sha,
    physical_inputs_sha256=sha(p/'physical_inputs.json'),sta_pass=False,board_download_allowed=False)
out=args.output.resolve();assert out.is_relative_to(ROOT/'output') and not out.exists()
out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f"PASS BINDING shared_RTL={len(common)} frame_cycles={fr['cycles']} actual_XLR_with_routing={used}/{capacity} NOT_TARGET_PASS")
