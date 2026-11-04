"""Freeze complete existing board plus the numerically qualified fixed compute RTL."""
import argparse,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,sha
from fixed_production_sources import sources
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--proof',type=Path,default=ROOT/'tmp/frame-execution/fixed-production-cluster1');p.add_argument('--production-proof',type=Path);p.add_argument('--base',type=Path,default=ROOT/'tmp/frame-execution/connected-board-taps-fixed1')
p.add_argument('--bridge-proof',type=Path);p.add_argument('--bridge-mapped-proof',type=Path);p.add_argument('--display-proof',type=Path)
p.add_argument('--fabric-proof',type=Path)
p.add_argument('--reset-proof',type=Path)
p.add_argument('--scaler-official-proof',type=Path);p.add_argument('--scaler-full-proof',type=Path);p.add_argument('--scaler-exhaustive-proof',type=Path)
p.add_argument('--raw-packer-proof',type=Path);p.add_argument('--raw-ingress-proof',type=Path);p.add_argument('--raw-front-proof',type=Path);p.add_argument('--raw-full-proof',type=Path)
p.add_argument('--map-mode',choices=('speed','area','cong'))
p.add_argument('--comb4-pack',type=int,choices=(0,1,2));a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
base=a.base.resolve();assert base.is_relative_to(ROOT/'tmp')
meta=json.loads((base/'inputs.json').read_text());prior=json.loads((base/'result.json').read_text())
assert sha(base/'inputs.json')==prior['inputs_sha256']
tested=a.proof.resolve()
proof=json.loads((tested/'result.json').read_text())
assert proof['status']=='FIXED_CLUSTER_REAL_RESIDENT_PASS'
assert sha(tested/'inputs.json')==proof['inputs_sha256']
replace={f.name:f for f in sources()}
qualified={Path(r['snapshot']).name:Path(r['snapshot']) for r in json.loads((tested/'inputs.json').read_text())['inputs']}
production_binding=None
if a.production_proof:
    pp=a.production_proof.resolve();pr=json.loads((pp/'result.json').read_text())
    assert pr['status']=='CONTINUOUS_FRAME_SMOKE_ONLY' and pr['commands']>=1550 and not pr['reasons']
    assert sha(pp/'inputs.json')==pr['inputs_sha256'] and sha(pp/'simulation.log')==pr['log_sha256']
    qp={}
    for row in json.loads((pp/'inputs.json').read_text())['inputs']:
        f=Path(row['snapshot']);assert sha(f)==row['sha256'];qp[f.name]=f
    for name in ('cnn_frame_datapath.sv','cnn_production_frame.sv','row_pair_dma.sv','frame_program_fetch.sv','frame_program_issue.sv'):
        if name in replace:qualified[name]=qp[name]
    production_binding=dict(path=str(pp),result_sha256=sha(pp/'result.json'),scope='1550 command production smoke, not full frame')
for name,f in replace.items():
    original=qualified[name].read_text();current=f.read_text()
    # The only permitted post-test delta removes debug-only hierarchical aliases
    # from synthesis; it leaves every simulated statement byte-identical.
    current=current.replace(' // synthesis translate_off\n','').replace(' // synthesis translate_on\n','')
    original=original.replace(' // synthesis translate_off\n','').replace(' // synthesis translate_on\n','')
    assert original==current,name
display_binding=None
if a.display_proof:
    gate=a.display_proof.resolve();gr=json.loads((gate/'result.json').read_text())
    assert gr['status']=='RGB720_DDR_FOURSTATE_PASS' and gr['display_burst']==128 and gr['row_stall']>0
    assert len(gr['lines'])==9 and sha(gate/'inputs.json')==gr['inputs_sha256'] and sha(gate/'simulation.log')==gr['log_sha256']
    gm=json.loads((gate/'inputs.json').read_text())
    for row in gm['inputs']:assert sha(row['snapshot'])==row['sha256']
    tb=(gate/'tb_rgb720_ddr.sv').read_text()
    assert 'WITHOUT resetting the reader' in tb and 'new frame admitted before unpacker empty' in tb
    for name in ('rgb24_stream_unpack.sv','rgb720_ddr_reader.sv'):replace[name]=gate/name
    display_binding=dict(path=str(gate),result_sha256=sha(gate/'result.json'),scope='Full 720p full/compare consecutive without reset; random stalls, row pauses, RRESP/RLAST and partial reset')
raw_binding=None
if any((a.raw_packer_proof,a.raw_ingress_proof,a.raw_front_proof,a.raw_full_proof)):
    assert all((a.raw_packer_proof,a.raw_ingress_proof,a.raw_front_proof,a.raw_full_proof))
    name='raw32_validated_packer.sv';candidate=None;bindings=[]
    # Manifest also retains unused vendor copies with duplicate basenames.
    # Bind only the paths actually compiled by the frozen board project.
    base_files={}
    for path in meta['source_files']:
        f=Path(path);assert f.name not in base_files,('Ambiguous active board source',f.name)
        base_files[f.name]=f
    for gate_arg,status in ((a.raw_packer_proof,'RAW_VALIDATED_PACKER_PASS'),
                           (a.raw_ingress_proof,'RAW_STREAM_INGRESS_PASS'),
                           (a.raw_front_proof,'RAW_STREAM_FRONTEND_PASS'),
                           (a.raw_full_proof,'RAW_STREAM_FULL_GEOMETRY_THREE_FRAME_PASS')):
        gate=gate_arg.resolve();gr=json.loads((gate/'result.json').read_text())
        assert gr['status']==status and sha(gate/'inputs.json')==gr['inputs_sha256']
        gm=json.loads((gate/'inputs.json').read_text())
        selected=None
        for row in gm['inputs']:
            f=Path(row['snapshot']);assert sha(f)==row['sha256']
            if f.name==name:selected=f
            elif not f.name.startswith('tb_'):
                assert f.name in base_files,('RAW dependency absent from board',f.name)
                compiled=base_files[f.name]
                assert sha(compiled)==row['sha256'] or compiled.read_text()==f.read_text(),('RAW dependency differs from board beyond newline encoding',f.name)
        assert selected is not None
        if candidate is not None:assert sha(selected)==sha(candidate)
        candidate=selected
        if 'results' in gr:
            assert len(gr['results'])==3
            for i,res in enumerate(gr['results']):assert sha(gate/f'simulation{i}.log')==res['log_sha256']
        else:assert sha(gate/'simulation.log')==gr['log_sha256']
        if status=='RAW_STREAM_FULL_GEOMETRY_THREE_FRAME_PASS':
            assert len(gr['frames'])==3 and 'good=3 aborted=0 compared_bytes=8294400' in gr['line']
        bindings.append(dict(path=str(gate),result_sha256=sha(gate/'result.json')))
    replace[name]=candidate
    raw_binding=dict(proofs=bindings,source_sha256=sha(candidate),scope='Official four-state poison/reset/hold and three full-geometry RAW-to-RGB frames with behavioral DDR')
bridge_binding=None
scaler_binding=None
if any((a.scaler_official_proof,a.scaler_full_proof,a.scaler_exhaustive_proof)):
    assert all((a.scaler_official_proof,a.scaler_full_proof,a.scaler_exhaustive_proof))
    scaler_name='rgb_bilinear_scaler_prototype.sv';bindings=[];candidate=None
    for gate_arg,status,full in (
        (a.scaler_official_proof,'RAW_ROW_UPSCALE_OFFICIAL_PIXEL_PASS',False),
        (a.scaler_full_proof,'RAW_ROW_UPSCALE_NATIVE_PIXEL_PASS',True),
        (a.scaler_exhaustive_proof,'EXHAUSTIVE_PRODUCTION_LERP_PASS',None)):
        gate=gate_arg.resolve();gr=json.loads((gate/'result.json').read_text())
        assert gr['status']==status and sha(gate/'inputs.json')==gr['inputs_sha256'] and sha(gate/'simulation.log')==gr['log_sha256']
        gm=json.loads((gate/'inputs.json').read_text())
        for row in gm['inputs']:assert sha(row['snapshot'])==row['sha256']
        if full is not None:
            assert gr['full']==full and not gr['reference'] and len(gr['lines'])==2
            assert 'BAD_MARKER_RESET_RECOVERY' in gr['lines'][1]
            if full:assert 'frames=3 pixels=6220800 width=1920 height=1080' in gr['lines'][0]
        else:assert gr['checks']==524288
        if candidate is not None:assert sha(candidate)==sha(gate/scaler_name)
        candidate=gate/scaler_name
        assert sha(gate/'resize_axis_up_2_to_3.sv')==sha(base/'rtl/resize_axis_up_2_to_3.sv')
        bindings.append(dict(path=str(gate),result_sha256=sha(gate/'result.json')))
    replace[scaler_name]=candidate
    scaler_binding=dict(proofs=bindings,source_sha256=sha(candidate),scope='Actual production lerp exhaustive; official RAM reset/stall/marker checks and three full-size independent pixel frames')
if a.bridge_proof:
    assert a.bridge_mapped_proof
    gate=a.bridge_proof.resolve();netgate=a.bridge_mapped_proof.resolve()
    gr=json.loads((gate/'result.json').read_text());nr=json.loads((netgate/'result.json').read_text())
    assert gr['status']==nr['status']=='DDR516_AXI_BRIDGE_PASS'
    bridge=gate/'ddr516_axi_address_bridge.sv'
    for folder,result in ((gate,gr),(netgate,nr)):
        assert sha(folder/'inputs.json')==result['inputs_sha256']
        gm=json.loads((folder/'inputs.json').read_text())
        for row in gm['inputs']:assert sha(row['snapshot'])==row['sha256']
        assert len(result['results'])==4 and result['late_bready'] and result['dependent_wready']
        for test in result['results']:assert sha(folder/f"seed{test['seed']}.log")==test['log_sha256']
        assert sha(result['cycle_equivalence_reference'])==sha(base/'ddr516_axi_address_bridge.sv')
    assert nr['mapped_public_primitives']
    ng=json.loads((netgate/'inputs.json').read_text())
    mapped=Path(next(row['source'] for row in ng['inputs'] if Path(row['snapshot']).name=='ddr516_axi_address_bridge.sv'))
    mr=json.loads((mapped.parent.parent/'result.json').read_text())
    assert mr['status']=='BURST_BUFFER_RAM_MAP_PASS' and mr['source_sha256']==sha(bridge) and mr['map_sha256']==sha(mapped)
    replace[bridge.name]=bridge
    bridge_binding=dict(rtl_gate=str(gate),rtl_result_sha256=sha(gate/'result.json'),mapped_gate=str(netgate),mapped_result_sha256=sha(netgate/'result.json'),scope='Same-cycle AXI protocol equivalence including valid stalled payload; controller and PHY unchanged')
fabric_binding=None
if a.fabric_proof:
    gate=a.fabric_proof.resolve();gr=json.loads((gate/'result.json').read_text());gm=json.loads((gate/'inputs.json').read_text())
    assert gr['status']=='AXI_RESPONSE_BROADCAST_CYCLE_EQ_PASS' and sha(gate/'inputs.json')==gr['inputs_sha256']
    assert len(gr['results'])==6
    for row in gm['inputs']:assert sha(row['snapshot'])==row['sha256']
    for test in gr['results']:
        assert sha(gate/test['log'])==test['log_sha256']
        assert any('PASS AXI_CYCLE_EQ' in line for line in test['lines'])
    for count in (3,4):
        assert any(test['arg']==f'-gN={count}' and any('max_burst=256' in line for line in test['lines']) for test in gr['results'])
    name='axi_three_master_burst_arbiter.sv'
    assert sha(base/'assembly'/name)==gr['reference_sha256']
    assert sha(gate/name)==gr['candidate_sha256']
    for name_dep in ('sapphire_axi32_to_axi128.sv','ti60_ddr_four_master_fabric.sv'):
        assert sha(gate/name_dep)==sha(base/'assembly'/name_dep)
    assert sha(gate/'ddr_memory_map.vh')==sha(base/'ddr_memory_map.vh')
    replace[name]=gate/name
    fabric_binding=dict(path=str(gate),result_sha256=sha(gate/'result.json'),scope=gr['scope'],candidate_sha256=gr['candidate_sha256'])
reset_binding=None
if a.reset_proof:
    gate=a.reset_proof.resolve();gr=json.loads((gate/'result.json').read_text());gm=json.loads((gate/'inputs.json').read_text())
    assert gr['status']=='RESET_TAIL_CYCLE_EQUIVALENCE_PASS' and len(gr['cases'])==4
    assert sha(gate/'inputs.json')==gr['inputs_sha256'] and sha(gate/'simulation.log')==gr['log_sha256']
    name='cnn_video_pipeline.sv';old=base/'assembly'/name;new=gate/name
    assert sha(old)==gr['base_sha256']==gm['base_sha256']
    assert sha(new)==gr['candidate_sha256']==gm['candidate_sha256']
    expected=old.read_text()
    for domain in ('process','display'):
        before=f'wire {domain}_rst_n = &{domain}_reset_pipe;'
        after=f'wire {domain}_rst_n = {domain}_reset_pipe[3];'
        assert expected.count(before)==1;expected=expected.replace(before,after)
    assert new.read_text()==expected
    replace[name]=new
    reset_binding=dict(path=str(gate),result_sha256=sha(gate/'result.json'),scope='Two last-stage reset outputs; four-state asynchronous assertion/four-edge release unchanged')
root.mkdir()
for folder in ('rtl','assembly'):shutil.copytree(base/folder,root/folder)
for f in base.iterdir():
    if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
records=[];seen=set()
for r in meta['inputs']:
    old=Path(r['snapshot']);assert sha(old)==r['sha256']
    src=replace.get(old.name,old);dst=root/old.relative_to(base)
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst),base_sha256=r['sha256']));seen.add(old.name)
paths=[root/Path(f).relative_to(base) for f in meta['source_files']]
for name,src in replace.items():
    if name in seen:continue
    dst=root/'assembly'/name;shutil.copy2(src,dst);paths.append(dst)
    records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
ns={'e':'http://www.efinixinc.com/enf_proj'};ET.register_namespace('efx',ns['e'])
xml=root/'ti60f225_oob.xml';tree=ET.parse(xml);design=tree.getroot().find('e:design_info',ns)
for e in list(design.findall('e:design_file',ns)):design.remove(e)
for i,f in enumerate(paths,1):design.insert(i,ET.Element('{'+ns['e']+'}design_file',name=f.relative_to(root).as_posix(),version='default',library='default'))
map_mode_change=None
if a.map_mode:
    mode=tree.getroot().find('e:synthesis/e:param[@name="mode"]',ns);assert mode is not None
    map_mode_change=dict(before=mode.get('value'),after=a.map_mode,scope='Synthesis objective only; same RTL, clocks, pins and timing constraints')
    mode.set('value',a.map_mode)
comb4_pack_change=None
if a.comb4_pack is not None:
    option=tree.getroot().find('e:synthesis/e:param[@name="pack-luts-to-comb4"]',ns)
    assert option is not None
    comb4_pack_change=dict(before=option.get('value'),after=a.comb4_pack,
                          scope='Official synthesis packing effort; same qualified RTL and timing constraints')
    option.set('value',str(a.comb4_pack))
ET.indent(tree,space='    ');tree.write(xml,encoding='utf-8',xml_declaration=False)
for r in records:
    if Path(r['snapshot'])==xml:r.update(source='fixed-window production source assembly',sha256=sha(xml))
meta.update(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),source_files=list(map(str,paths)),
            fixed_cluster_proof_sha256=sha(tested/'result.json'),production_binding=production_binding,bridge_binding=bridge_binding,display_binding=display_binding,scaler_binding=scaler_binding,raw_binding=raw_binding,fabric_binding=fabric_binding,reset_binding=reset_binding or meta.get('reset_binding'),map_mode_change=map_mode_change,comb4_pack_change=comb4_pack_change,runner_sha256=sha(__file__),board_release=False)
(root/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n')
for r in records:assert sha(r['snapshot'])==r['sha256']
print('FIXED_PRODUCTION_BOARD_FROZEN',len(records))
