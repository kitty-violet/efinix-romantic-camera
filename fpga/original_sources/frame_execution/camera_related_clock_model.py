"""Keep PLL4's exact 4:12:120 ratios at the existing tighter 6.666ns anchor.

Physical PLL configuration is unchanged. This does not cut any path or relax
any period: 50MHz/5MHz analysis periods tighten by the same 0.01% as 150MHz.
"""
import json, re, shutil, subprocess, time, xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT, HERE, EF, sha
from windows_process_tree import stop_launched_tree

def transform(sdc, peri):
    tree=ET.fromstring(peri);ns={'e':'http://www.efinixinc.com/peri_design'}
    pll=next(x for x in tree.iter() if x.tag.endswith('}pll') and x.get('name')=='pll_inst4')
    outs={x.get('name'):x for x in pll if x.tag.endswith('}comp_output_clock')}
    for name,div in (('process_clk_150',4),('i_mipi_rx_pclk',12),('CLK_5M',120)):
        assert outs[name].get('out_divider')==str(div)
        assert outs[name].get('phase_setting')=='0' and outs[name].get('is_inverted')=='false'
    assert 'create_clock -period 6.666 -name process_clk_150 [get_ports {process_clk_150}]' in sdc
    changes=[]
    for name,before,after in (('i_mipi_rx_pclk','20.000','19.998'),('CLK_5M','200.000','199.980')):
        old=f'create_clock -period {before} -name {name} [get_ports {{{name}}}]'
        new=f'create_clock -period {after} -name {name} [get_ports {{{name}}}]'
        assert sdc.count(old)==1;sdc=sdc.replace(old,new)
        changes.append(dict(clock=name,before_ns=float(before),after_ns=float(after)))
    return sdc,changes

def main():
    base=ROOT/'tmp/frame-execution/camera-checked-bound2'
    out=ROOT/'tmp/frame-execution/camera-related-clock-sta1'
    assert not out.exists();out.mkdir()
    sdc,changes=transform((base/'ti60f225_oob.sdc').read_text(),(base/'ti60f225_oob.peri.xml').read_text())
    frozen={str(base/p):sha(base/p) for p in ('ti60f225_oob.sdc','ti60f225_oob.peri.xml',
        'outflow/ti60f225_oob.troutingtraces','outflow/ti60f225_oob.place','outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb')}
    for folder in ('outflow','work_pnr'):shutil.copytree(base/folder,out/folder)
    for f in base.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'):shutil.copy2(f,out/f.name)
    (out/'ti60f225_oob.sdc').write_text(sdc,encoding='utf8')
    tcl='report_timing_summary -file related_summary.rpt\ncheck_timing -file related_checks.rpt -verbose\n'
    for name,source,target in (('cfg','CLK_5M','process_clk_150'),('pixel','i_mipi_rx_pclk','process_clk_150'),('core','process_clk_150','process_clk_150')):
        tcl+=f'report_timing -setup -from_clock {{{source}}} -to_clock {{{target}}} -npaths 128 -nworst 1 -file related_{name}.rpt\n'
    (out/'related.tcl').write_text(tcl)
    (out/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script related.tcl --timeout 180\nexit /b %errorlevel%\n')
    (out/'inputs.json').write_text(json.dumps(dict(original=frozen,changes=changes,modeler_sha256=sha(__file__),
        physical_clock_changed=False,new_exceptions=0,hardware_access=False),indent=2))
    started=time.monotonic()
    with (out/'sta.log').open('w') as log:
        child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=out,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=210)
        except subprocess.TimeoutExpired:
            (out/'timeout.json').write_text(json.dumps(stop_launched_tree(child)));raise
    assert code==0
    for f,h in frozen.items():assert sha(f)==h
    text=(out/'sta.log').read_text()+'\n'+(out/'outflow/ti60f225_oob.sta.out').read_text()
    assert not re.search(r'Unable to run|No (?:ports|pins|clocks) matched|Timer cuts a combinational loop|^\s*(?:ERROR|FATAL)\s*:',text,re.M)
    results={}
    for name in ('cfg','pixel','core'):
        p=out/'outflow'/f'related_{name}.rpt';body=p.read_text()
        slacks=[float(x) for x in re.findall(r'^Slack\s*:\s*([\d.-]+)',body,re.M)]
        assert slacks,name;results[name]=dict(worst_ns=min(slacks),paths=len(slacks),report_sha256=sha(p))
    (out/'result.json').write_text(json.dumps(dict(status='RELATED_CLOCK_SAME_ROUTE_DIAGNOSTIC',results=results,
        seconds=time.monotonic()-started,inputs_sha256=sha(out/'inputs.json'),board_release=False,hardware_access=False),indent=2))
    print(json.dumps(results))

if __name__=='__main__':main()
