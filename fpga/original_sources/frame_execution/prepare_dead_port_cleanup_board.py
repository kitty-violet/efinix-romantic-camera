"""Remove proven nonfunctional legacy core ports, preserving physical I/O/PLL.

DDR5.16 uses periphery CLKOUT; legacy CK hi/lo are constants with no pin.
The selected Sapphire JTAG bridge does not consume RESET/UPDATE (confirmed
in the complete mapped design). Keep all active JTAG and reset connections.
"""
import argparse,copy,json,re,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,sha
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
base=a.base.resolve();proof=a.proof.resolve();root=a.root.resolve()
assert root.is_relative_to(ROOT/'tmp') and not root.exists()
meta=json.loads((base/'inputs.json').read_text());pr=json.loads((proof/'result.json').read_text())
assert sha(proof/'inputs.json')==pr['inputs_sha256']
net=proof/'outflow/ti60f225_oob.map.v';assert sha(net)==pr['map_sha256']
body=net.read_text()
# Eliminate declarations/module signature/comments; no remaining functional use.
functional=body[body.index(');')+2:]
functional=re.sub(r'/\*.*?\*/|//[^\n]*','',functional,flags=re.S)
for name in ('jtag_inst1_RESET','jtag_inst1_UPDATE'):
    stripped=re.sub(r'\binput\s+'+name+r'\s*;','',functional)
    assert not re.search(r'\b'+name+r'\b',stripped),name
top=Path(next(x['snapshot'] for x in meta['inputs'] if Path(x['snapshot']).name=='ti60f225_oob_top.v'))
proved_top=proof/'rtl/ti60f225_oob_top.v'
assert sha(top)==sha(proved_top),'Proof must use the same complete top'
controller=base/'ddr516_v4_controller.sv';assert 'assign ddr_ck_hi=0;assign ddr_ck_lo=1;' in controller.read_text()
text=top.read_text()
for name in ('jtag_inst1_RESET','jtag_inst1_UPDATE','ddr_ck_hi','ddr_ck_lo'):
    pattern=r'^.*\b(?:input|output)\s+'+name+r',\s*\n'
    text,n=re.subn(pattern,'',text,flags=re.M);assert n==1,(name,n)
# JTAG values are irrelevant to all observable mapped logic; CK constants
# remain internal wires for wrapper compatibility, not unlocated core outputs.
index=text.index(');')+2
text=text[:index]+"\n wire jtag_inst1_RESET=1'b0,jtag_inst1_UPDATE=1'b0;\n wire ddr_ck_hi,ddr_ck_lo;\n"+text[index:]
peri=base/'ti60f225_oob.peri.xml';tree=ET.parse(peri);ns={'p':'http://www.efinixinc.com/peri_design_db'}
ck=tree.find('.//p:comp_gpio[@name="ddr_ck"]',ns)
assert ck is not None and ck.attrib['mode']=='clkout' and ck.attrib['gpio_def']=='GPIOR_P_09'
clock=ck.find('p:output_config',ns)
assert clock.attrib['clock_name']=='sdram_clk' and clock.attrib['is_clock_inverted']=='true'
protected={tag:ET.tostring(tree.find('p:'+tag,ns)) for tag in ('gpio_info','pll_info','lvds_info','mipi_dphy_info')}
gen=tree.find('.//p:jtag[@name="jtag_inst1"]/p:gen_pin',ns);assert gen is not None
removed=[]
for pin in list(gen):
    if pin.attrib['name'] in ('jtag_inst1_RESET','jtag_inst1_UPDATE'):removed.append(pin.attrib['name']);gen.remove(pin)
assert len(removed)==2
assert all(ET.tostring(tree.find('p:'+tag,ns))==v for tag,v in protected.items())
sdc=base/'ti60f225_oob.sdc';kept=[];deleted=[]
for line in sdc.read_text().splitlines():
    if re.search(r'\b(jtag_inst1_RESET|jtag_inst1_UPDATE|ddr_ck_hi|ddr_ck_lo)\b',line):
        assert line.startswith(('set_input_delay ','set_output_delay '));deleted.append(line)
    else:kept.append(line)
assert len(deleted)==6
root.mkdir();records=[]
for folder in ('rtl','assembly'):shutil.copytree(base/folder,root/folder)
for row in meta['inputs']:
    f=Path(row['snapshot']);assert sha(f)==row['sha256'];dst=root/f.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(f,dst)
    if f==top:dst.write_text(text)
    if f==sdc:dst.write_text('\n'.join(kept)+'\n')
    if f==peri:tree.write(dst,encoding='utf-8',xml_declaration=True)
    records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(dst),base_sha256=row['sha256']))
assert sum(x['sha256']!=x['base_sha256'] for x in records)==3
binding=dict(proof=str(proof),map_sha256=pr['map_sha256'],unused_jtag_inputs=removed,
             obsolete_core_ck_outputs=['ddr_ck_hi','ddr_ck_lo'],deleted_constraints=deleted,
             gpio_pll_lvds_mipi_unchanged=True,physical_pin_change=False,clock_change=False)
warning_base=str(proof)
meta.update(inputs=records,base=str(base),warning_reference_base=warning_base,base_inputs_sha256=sha(base/'inputs.json'),
            source_files=[str(root/Path(f).relative_to(base)) for f in meta['source_files']],
            dead_port_cleanup=binding,runner_sha256=sha(__file__),board_release=False)
(root/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n');print('DEAD_PORT_CLEANUP_BOARD_FROZEN',len(records),flush=True)
