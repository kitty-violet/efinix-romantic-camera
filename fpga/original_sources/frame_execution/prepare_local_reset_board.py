"""Isolated full-board reset delta, with freshly synthesized netlist. No USB."""
import argparse
import collections
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
from run_units import ROOT,HERE,EF,sha
from board_measured_pinmap import audit as audit_parent
from windows_process_tree import stop_launched_tree

BASE=ROOT/'tmp/frame-execution/connected-board-measured-pinmap1'
VARIANT=HERE/'variants/board_deterministic_reset/board_startup_reset.sv'
TOP='rtl/ti60f225_oob_top.v'
OLD='assign w_arstn  \t\t       = i_pll_locked & pll_locked & user_pll_locked;'
NEW='''// Direct manual assertion and deterministic configuration startup.
board_startup_reset startup_reset (
 .clk(process_clk_150), .key_reset_n(i_arstn),
 .plls_locked(i_pll_locked & pll_locked & user_pll_locked & process_pll_locked),
 .system_reset_n(w_arstn)
);
wire w_i2c_arstn;
board_startup_reset startup_reset_5m (
 .clk(CLK_5M), .key_reset_n(i_arstn),
 .plls_locked(i_pll_locked & pll_locked & user_pll_locked & process_pll_locked),
 .system_reset_n(w_i2c_arstn)
);'''


def patched_top(text):
    assert text.count(OLD)==1
    text=text.replace(OLD,NEW).replace('    input wire process_clk_150,', '    input wire process_clk_150,\n    input wire process_pll_locked,')
    block='always @( posedge CLK_5M or negedge w_arstn )\nbegin\n    if( !w_arstn )'
    assert text.count(block)==1
    return text.replace(block,block.replace('w_arstn','w_i2c_arstn'))


def audit_reset_periphery(board):
    from board_measured_pinmap import signature, pin_audit
    pin_audit(BASE/'ti60f225_oob.peri.xml')
    before=ET.parse(BASE/'ti60f225_oob.peri.xml').getroot()
    after=ET.parse(board/'ti60f225_oob.peri.xml').getroot()
    pll=next(n for n in after.iter() if n.tag.endswith('}pll') and n.get('name')=='pll_inst4')
    assert pll.get('locked_name')=='process_pll_locked'
    pll.set('locked_name','')
    assert signature(after)==signature(before),'Unexpected peripheral change beyond LOCKED signal'
    old=(BASE/'ti60f225_oob.sdc').read_text()
    expected=old.replace('i_arstn i_pll_locked pll_locked user_pll_locked',
                         'i_arstn i_pll_locked pll_locked user_pll_locked process_pll_locked')
    assert (board/'ti60f225_oob.sdc').read_text()==expected
    return dict(status='FOUR_PLL_LOCK_ONLY_PERIPHERY_DELTA_PASS',
                all_physical_pins_and_frequency_phase_unchanged=True,
                new_exception_scope='Only new asynchronous PLL LOCKED input; no existing timing requirement changed')


def audit_delta(board):
    meta=json.loads((board/'inputs.json').read_text())
    for r in meta['inputs']:assert sha(r['snapshot'])==r['sha256']
    old=(BASE/TOP).read_text();new=(board/TOP).read_text()
    assert old.count(OLD)==1 and new==patched_top(old)
    assert sha(board/'assembly/board_startup_reset.sv')==sha(VARIANT)
    xml=(board/'ti60f225_oob.xml').read_text()
    added='        <efx:design_file name="assembly/board_startup_reset.sv" version="default" library="default" />\n'
    assert xml.count(added)==1 and xml.replace(added,'')==(BASE/'ti60f225_oob.xml').read_text()
    changed=[]
    for r in meta['inputs']:
        f=Path(r['snapshot']);rel=f.relative_to(board)
        if not (BASE/rel).exists() or sha(f)!=sha(BASE/rel):changed.append(rel.as_posix())
    assert set(changed)=={TOP,'assembly/board_startup_reset.sv','ti60f225_oob.xml','ti60f225_oob.peri.xml','ti60f225_oob.sdc'},changed
    audit_reset_periphery(board)
    return dict(changed=changed,all_other_inputs_unchanged=True,
                pins_ddr_frequencies_unchanged=True,only_new_async_lock_port_classified=True,hardware_access=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--short-proof',required=True,type=Path);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    parent=audit_parent(BASE)
    proof=a.short_proof.resolve();short=json.loads((proof/'result.json').read_text())
    assert short['status']=='RESET_AND_ACTUAL_A_STORE_FOURSTATE_PASS'
    assert short['inputs_sha256']==sha(proof/'inputs.json')
    sm=json.loads((proof/'inputs.json').read_text())
    for r in sm['inputs']:assert sha(r['source'])==sha(r['snapshot'])==r['sha256']
    root.mkdir()
    for name in ('rtl','assembly'):shutil.copytree(BASE/name,root/name)
    for f in BASE.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
    old=(root/TOP).read_text();assert old.count(OLD)==1
    (root/TOP).write_text(patched_top(old),encoding='utf-8')
    peri=root/'ti60f225_oob.peri.xml';pt=ET.parse(peri)
    pll=next(n for n in pt.iter() if n.tag.endswith('}pll') and n.get('name')=='pll_inst4')
    assert pll.get('locked_name')=='';pll.set('locked_name','process_pll_locked')
    ET.register_namespace('efxpt','http://www.efinixinc.com/peri_design_db')
    pt.write(peri,encoding='utf-8',xml_declaration=True)
    sdc=root/'ti60f225_oob.sdc';st=sdc.read_text()
    oldlock='i_arstn i_pll_locked pll_locked user_pll_locked'
    assert st.count(oldlock)==1
    sdc.write_text(st.replace(oldlock,oldlock+' process_pll_locked'),encoding='utf-8')
    shutil.copy2(VARIANT,root/'assembly/board_startup_reset.sv')
    xml=root/'ti60f225_oob.xml';body=xml.read_text();needle='        <efx:top_module name="ti60f225_oob_top" />\n'
    assert body.count(needle)==1
    xml.write_text(body.replace(needle,needle+'        <efx:design_file name="assembly/board_startup_reset.sv" version="default" library="default" />\n'),encoding='utf-8')
    meta=json.loads((BASE/'inputs.json').read_text())
    for r in meta['inputs']:
        rel=Path(r['snapshot']).relative_to(BASE);r['snapshot']=str(root/rel);r['sha256']=sha(root/rel)
    meta['inputs'].append(dict(source=str(VARIANT),snapshot=str(root/'assembly/board_startup_reset.sv'),sha256=sha(VARIANT)))
    meta.update(parent=str(BASE),parent_audit=parent,reset_short_proof=str(proof),
                reset_short_sha256=sha(proof/'result.json'),reset_runner_sha256=sha(__file__),
                board_download_allowed=False,hardware_access=False)
    (root/'inputs.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    delta=audit_delta(root)
    (root/'reset_delta_audit.json').write_text(json.dumps(delta,indent=2))
    (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow map --timeout 1200\nexit /b %errorlevel%\n')
    try:
        import sys
        subprocess.run([sys.executable,str(HERE/'validate_local_reset_binding.py'),'--board',str(root),
                        '--root',str(root/'local-reset-short')],cwd=ROOT,check=True,timeout=120)
        start=time.monotonic()
        with (root/'map.log').open('w') as out:
            child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=root,stdout=out,stderr=subprocess.STDOUT)
            try:code=child.wait(timeout=1260)
            except subprocess.TimeoutExpired:
                (root/'timeout_cleanup.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
        body=(root/'map.log').read_text(errors='replace')
        assert code==0 and re.search(r'map\s*:\s*PASS',body),(code,body[-3000:])
        warning=root/'outflow/ti60f225_oob.warn.log';ws=warning.read_text(errors='replace')
        pattern=r'^\[EFX-0201 WARNING\] (.*?)\. \('
        prior=(BASE/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
        assert not set(re.findall(pattern,ws,re.M))-set(re.findall(pattern,prior,re.M))
        assert 'EFX-0654' not in ws
        net=root/'outflow/ti60f225_oob.map.v';nettext=net.read_text()
        rpt=root/'outflow/ti60f225_oob.map.rpt'
        counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
        dumped=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',nettext,re.M)))
        for k in ('RAM10','DPRAM10','DSP24','DSP48'):assert counts[k]==dumped[k]
        assert 'startup_reset' in nettext and 'lock_pipe' in nettext
        audit_delta(root)
        result=dict(status='DETERMINISTIC_RESET_FULL_BOARD_MAP_NOT_RELEASE',resources=counts,
                    dumped_net_primitives=dumped,seconds=time.monotonic()-start,
                    inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),
                    report_sha256=sha(rpt),warning_sha256=sha(warning),comb_loop_warning=False,
                    physical_timing_pass=False,board_download_allowed=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()


