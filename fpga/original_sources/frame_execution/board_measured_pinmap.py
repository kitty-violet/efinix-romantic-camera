"""Freeze and audit five evidence-based GPIO corrections; never access hardware."""
import argparse, copy, json, shutil, xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT, HERE, sha
from audit_bringup_observation import audit as audit_parent

BASE=ROOT/'tmp/frame-execution/connected-board-bringup-status2'
VARIANT=HERE/'variants/board_measured_pinmap/ti60f225_oob.peri.xml'
KEY_PROOF=ROOT/'output/toolchain/实板按键映射_2026-10-09.json'
EXAMPLE=ROOT/'官方示例/01_Ti60F225_Key_led_osc_demo/01_Ti60F225_key_demo/LED_8bit_Test.peri.xml'
VIDEO=ROOT/'tmp/efinity-smoke-2026.1/sc431-v6/ti60f225_oob.peri.xml'
SOC=ROOT/'官方示例/08_ti60f225_soc_demo/09_Ti60F225_hardjtag_demo/par/ddr_demo_ti60/ddr_demo_ti60.peri.xml'
MANUAL=ROOT/'output/toolchain/project_evidence_audit_2026-09-28/pdf_text/TI60_TI60F225I3-V4 DEMO板软硬件设计说明-260813-1_cd209e9e2e7a.txt'
NS={'p':'http://www.efinixinc.com/peri_design_db'}
CHANGES={'i_arstn':('GPIOR_27','GPIOR_22','3.3 V LVCMOS'),
 'led[0]':('GPIOR_21','GPIOR_27','3.3 V LVCMOS'),
 'led[1]':('GPIOR_22','GPIOR_24','3.3 V LVCMOS'),
 'led[2]':('GPIOR_28','GPIOL_P_12','1.8 V LVCMOS'),
 'led[3]':('GPIOR_25','GPIOL_P_02','1.8 V LVCMOS')}

def nodes(tree,kind):
    return {n.get('name'):n for n in tree.findall('.//p:'+kind,NS)}
def signature(n):
    return (n.tag, sorted(n.attrib.items()), (n.text or '').strip(), [signature(c) for c in n])

def pin_audit(peri):
    proof=json.loads(KEY_PROOF.read_text(encoding='utf-8'))
    assert proof['status']=='PASS_KEY_MAPPING_ONLY' and proof['all_keys_released']
    assert [k['ball'] for k in proof['keys']]==['C4','P14','N14','A3']
    for receipt in proof['receipts']: assert sha(ROOT/receipt['source'])==receipt['sha256']
    old=ET.parse(BASE/'ti60f225_oob.peri.xml').getroot(); new=ET.parse(peri).getroot()
    ref=ET.parse(EXAMPLE).getroot();video=ET.parse(VIDEO).getroot();soc=ET.parse(SOC).getroot()
    oldg,newg,refg=nodes(old,'comp_gpio'),nodes(new,'comp_gpio'),nodes(ref,'comp_gpio')
    assert set(oldg)==set(newg)
    delta=[]
    for name,(before,after,voltage) in CHANGES.items():
        a,b=oldg[name],newg[name]
        assert a.get('gpio_def')==before and b.get('gpio_def')==after
        assert b.get('io_standard')==voltage
        expected=copy.deepcopy(a);expected.set('gpio_def',after);expected.set('io_standard',voltage)
        assert signature(expected)==signature(b),(name,'unexpected change')
        refname='key_i[1]' if name=='i_arstn' else name.replace('led[','led_data[')
        assert b.get('gpio_def')==refg[refname].get('gpio_def')
        assert b.get('io_standard')==refg[refname].get('io_standard')
        assert b.get('mode')==refg[refname].get('mode')
        delta.append(dict(name=name,before=before,after=after,io_standard=voltage))
    normalized=copy.deepcopy(new)
    for name,node in nodes(normalized,'comp_gpio').items():
        if name in CHANGES: node.attrib=copy.deepcopy(oldg[name].attrib)
    assert signature(normalized)==signature(old),'Change outside five GPIO attributes'
    pins=[n.get('gpio_def') for n in newg.values()]
    assert len(pins)==len(set(pins)),'Duplicate GPIO allocation'
    key_gpios={k['gpio'] for k in proof['keys']}
    assert not [n.get('name') for n in newg.values() if n.get('gpio_def') in key_gpios and n.get('mode')!='input']
    assert new.find('.//p:global_unused_config',NS).get('state')=='input with weak pullup'
    # Audit all remaining external interfaces, not just the corrected key.
    vg,sg=nodes(video,'comp_gpio'),nodes(soc,'comp_gpio')
    reviewed=set(CHANGES)
    for name,n in newg.items():
        if name.startswith('ddr_') or name in ('verf0','verf1'):
            for refnodes in (vg,sg):
                assert all(n.get(k)==refnodes[name].get(k) for k in ('gpio_def','io_standard')),name
                if name=='ddr_ck':
                    # 5.16 uses a clock-output primitive instead of old DDR output data.
                    # Physical pin/voltage are identical; implementation was already
                    # validated in the parent vendor short test and is unchanged here.
                    assert n.get('mode')=='clkout' and refnodes[name].get('mode')=='output'
                else: assert n.get('mode')==refnodes[name].get('mode'),name
            reviewed.add(name)
    for name,gpio in {'CLK_25M':'GPIOL_P_08','CLK_27M':'GPIOL_P_10',
                      'io_cam_scl':'GPIOR_20','io_cam_sda':'GPIOR_18','o_cam_rst_p':'GPIOR_16'}.items():
        assert newg[name].get('gpio_def')==vg[name].get('gpio_def')==gpio
        assert newg[name].get('io_standard')==vg[name].get('io_standard')
        reviewed.add(name)
    assert reviewed==set(newg),('Unreviewed GPIO',set(newg)-reviewed)
    for name,block in {'cam_ck':'GPIOT_PN_09','cam_d0':'GPIOT_PN_07','cam_d1':'GPIOT_PN_10',
                       'cam_d2':'GPIOT_PN_06','cam_d3':'GPIOT_PN_11'}.items():
        n=nodes(new,'mipi_dphy')[name]
        assert n.get('block_def')==nodes(video,'mipi_dphy')[name].get('block_def')==block
        assert n.get('ops_type')=='rx'
    for name,block in {'tmds_tx_clk':'GPIOR_PN_10','tmds_tx_data0':'GPIOR_PN_13',
                       'tmds_tx_data1':'GPIOR_PN_11','tmds_tx_data2':'GPIOR_PN_12'}.items():
        n=nodes(new,'lvds')[name]
        assert n.get('lvds_def')==nodes(video,'lvds')[name].get('lvds_def')==block
        assert n.get('ops_type')=='tx'
    banks=nodes(new,'iobank')
    assert banks['1A'].get('iostd')==banks['1B'].get('iostd')=='1.8 V LVCMOS'
    assert banks['2A'].get('iostd')==banks['2B'].get('iostd')=='1.2 V LVCMOS'
    return delta

def audit(board):
    board=board.resolve(); parent=audit_parent(BASE)
    meta=json.loads((board/'inputs.json').read_text());mapped=json.loads((board/'result.json').read_text())
    assert sha(board/'inputs.json')==mapped['inputs_sha256']
    changed=[]
    for row in meta['inputs']:
        p=Path(row['snapshot']);assert p.is_relative_to(board) and sha(p)==row['sha256']
        rel=p.relative_to(board)
        if sha(p)!=sha(BASE/rel):changed.append(rel.as_posix())
    assert changed==['ti60f225_oob.peri.xml'],changed
    delta=pin_audit(board/'ti60f225_oob.peri.xml')
    net=board/'outflow/ti60f225_oob.map.v'
    assert sha(net)==parent['map_sha256']==mapped['map_sha256']
    # Every elaboration input except physical GPIO attributes is byte-identical.
    # Reuse the actual synthesis result, without claiming a new synthesis run.
    return dict(status='MEASURED_GPIO_DELTA_SAME_RTL_MAP_PASS',board=str(board),parent=str(BASE),
                changes=delta,map_sha256=sha(net),inputs_sha256=sha(board/'inputs.json'),
                parent_observation=parent,rtl_and_constraints_unchanged=True,
                pins_changed=True,physical_timing_pass=False,hardware_access=False)

def prepare(board):
    assert board.is_relative_to(ROOT/'tmp') and not board.exists()
    parent=audit_parent(BASE);pin_audit(VARIANT)
    board.mkdir()
    for name in ('rtl','assembly','outflow','work_syn'):shutil.copytree(BASE/name,board/name)
    for f in BASE.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,board/f.name)
    shutil.copy2(VARIANT,board/VARIANT.name)
    meta=json.loads((BASE/'inputs.json').read_text())
    for row in meta['inputs']:
        rel=Path(row['snapshot']).relative_to(BASE);row['snapshot']=str(board/rel)
        row['sha256']=sha(board/rel)
        if rel.as_posix()=='ti60f225_oob.peri.xml':row['source']=str(VARIANT)
    meta['measured_pinmap']={str(p):sha(p) for p in (KEY_PROOF,EXAMPLE,VIDEO,SOC,MANUAL,VARIANT,Path(__file__))}
    meta['board_download_allowed']=False
    (board/'inputs.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    mapped=json.loads((BASE/'result.json').read_text())
    mapped.update(status='PINMAP_CHANGED_REUSED_PARENT_MAP_NOT_RELEASE',seconds=0,
                  synthesis_reused_from=str(BASE),inputs_sha256=sha(board/'inputs.json'),
                  physical_timing_pass=False,board_download_allowed=False)
    (board/'result.json').write_text(json.dumps(mapped,indent=2),encoding='utf-8')
    result=audit(board)
    (board/'pinmap_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
    prepare(a.board.resolve())
