"""Read-only DDR startup contract audit of an existing frozen board source.

 Checks declared digital configuration; does not certify physical calibration.
 All reports go to a new directory, and frozen inputs are hash bound.
"""
import argparse, json, re, xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT, EF, sha

PNS = {'p':'http://www.efinixinc.com/peri_design_db'}

def read_rom(root):
    header=(root/'ddr3_controller.vh').read_text()
    labels={k:int(v) for k,v in re.findall(r"`define (RAM_\w+) 9'd(\d+)",header)}
    words=[int(v,16) for v in (root/'ddr3_controller.bin').read_text().splitlines()]
    return header,labels,words

def audit(board):
    records={}
    def bind(path):
        path=Path(path);records[str(path)]=sha(path);return path
    tree=ET.parse(bind(board/'ti60f225_oob.peri.xml'))
    gpios={x.get('name'):x for x in tree.findall('.//p:comp_gpio',PNS)}
    pll=next(x for x in tree.findall('.//p:pll',PNS) if x.get('name')=='DDR3_PLL')
    clocks={x.get('name'):x for x in pll.findall('p:comp_output_clock',PNS)}
    fb=pll.find('p:adv_prop',PNS).get('feedback_clock_name')
    fin=float(pll.get('ref_clock_freq'));m=int(pll.get('multiplier'));n=int(pll.get('pre_divider'));o=int(pll.get('post_divider'))
    fpll=fin*m*int(clocks[fb].get('out_divider'))/n
    freq={name:fpll/int(x.get('out_divider')) for name,x in clocks.items()}
    expected={'tx_cal_clk_90edge':400.,'sdram_clk':400.,'rx_cal_clk':400.,'ddr_core_clk200':200.,'ddr_feedback50':50.}
    assert freq==expected,(freq,expected)
    assert 2200<=fpll*o<=5500 and fpll<=4000
    assert clocks['rx_cal_clk'].get('is_dyn_phase')=='true'
    assert clocks['rx_cal_clk'].get('number')=='2'
    assert clocks['tx_cal_clk_90edge'].get('phase_setting')=='3'
    assert all(clocks[k].get('phase_setting')=='0' for k in expected if k!='tx_cal_clk_90edge')
    pin_checks=[]
    for name,node in gpios.items():
        if not name.startswith('ddr_'):continue
        cfg=node.find('p:output_config',PNS)
        assert node.get('io_standard')==('1.5 V Differential SSTL' if name=='ddr_ck' or name.startswith('ddr_dqs[') else '1.5 V SSTL'),(name,node.attrib)
        if name=='ddr_ck':
            assert node.get('mode')=='clkout' and cfg.get('clock_name')=='sdram_clk'
            assert cfg.get('is_clock_inverted')=='true'
        else:
            is_dq=name.startswith('ddr_dq[');is_dqs=name.startswith('ddr_dqs[');is_dm=name.startswith('ddr_dm[')
            outclk='tx_cal_clk_90edge' if is_dq or is_dm else 'sdram_clk'
            assert cfg.get('clock_name')==outclk,(name,cfg.attrib)
            assert cfg.get('register_option')=='register' and cfg.get('is_clock_inverted')=='false'
            assert cfg.get('ddio_type')==('resync' if is_dq or is_dqs or is_dm else 'none')
            if is_dq or is_dqs:
                inc=node.find('p:input_config',PNS);oe=node.find('p:output_enable_config',PNS)
                assert inc.get('is_register')=='true' and inc.get('ddio_type')=='resync'
                assert inc.get('clock_name')=='rx_cal_clk' and inc.get('is_clock_inverted')=='false'
                assert oe.get('is_register')=='true' and oe.get('clock_name')==outclk
        pin_checks.append({'signal':name,'pad':node.get('gpio_def'),'output_clock':cfg.get('clock_name')})
    original=ROOT/'tmp/frame-execution/ddr516-generated2/ip/ddr516_reference'
    guarded=ROOT/'tmp/frame-execution/ddr516-read-pre-guard1/ip/ddr516_reference'
    for folder in (original,guarded,board):
        bind(folder/'ddr3_controller.vh');bind(folder/'ddr3_controller.bin')
    h,labels,words=read_rom(board);_,old_labels,old_words=read_rom(original)
    assert sha(board/'ddr3_controller.bin')==sha(guarded/'ddr3_controller.bin')
    assert sha(board/'ddr3_controller.vh')==sha(guarded/'ddr3_controller.vh')
    changes=[];keys=sorted(old_labels,key=old_labels.get)
    for i,key in enumerate(keys):
        nxt=keys[i+1] if i+1<len(keys) else None
        before=old_words[old_labels[key]:old_labels[nxt] if nxt else len(old_words)]
        after=words[labels[key]:labels[nxt] if nxt else len(words)]
        if before!=after:
            assert key in ('RAM_RD2PREA','RAM_RPAW','RAM_RPAR')
            assert after[:3]==[int('3e00000003e0000000',16)]*3 and after[3:]==before
            changes.append(key)
    assert set(changes)=={'RAM_RD2PREA','RAM_RPAW','RAM_RPAR'}
    init=words[labels['RAM_INIT']:labels['RAM_WAITING']];mrs=[]
    for i,word in enumerate(init):
        for half,cmd in enumerate((word>>36,word&((1<<36)-1))):
            if (cmd>>29)&0xf==0 and (cmd>>33)&1:
                mrs.append({'word':labels['RAM_INIT']+i,'half':half,'mr':(cmd>>25)&7,'value':(cmd>>9)&0xffff})
    assert [(x['mr'],x['value']) for x in mrs]==[(2,0),(3,0),(1,0x44),(0,0x1520)],mrs
    params=json.loads(bind(guarded.parent.parent/'generation_request.json').read_text())['resolved']
    for k,v in {'data_width':'1','row':'14','col':'10','bank':'3','freq':'400','simulation':"1'b0",'tCL':'6','tCWL':'5','usReset':'200','usCKE':'500','PLL_OUT_SEL':'4'}.items():assert params[k]==v,(k,params[k])
    # Compilation inputs must contain only the current controller and wrapper.
    p=ET.parse(bind(board/'ti60f225_oob.xml'));ns={'e':'http://www.efinixinc.com/enf_proj'}
    files=[x.get('name') for x in p.findall('.//e:design_file',ns)]
    assert 'ddr516_reference.sv' in files and not any('rtl/ddr3_controller/' in x and x.endswith(('.v','.sv')) for x in files)
    top=bind(board/'rtl/ti60f225_oob_top.v').read_text();wrap=bind(board/'ddr516_v4_controller.sv').read_text()
    compact=lambda s:re.sub(r'\s+','',s)
    for token in ('.axi_clk(process_clk_150)', '.core_clk(ddr_core_clk200)', '.tx_cal_clk(sdram_clk)', '.rstn(w_arstn)'):assert token in compact(top),token
    for token in ('.cal_ena(1\'b1)', 'ready_meta<=calibration_done&&calibration_pass;', '.reset_n(rstn&&ready_sync)', '.tac_clk(rx_cal_clk)', '.tdqss_clk(tx_cal_clk)'):assert token in compact(wrap),token
    bind(board/'ddr516_rx_retime.sv');bind(board/'assembly/board_startup_reset.sv')
    # The original vendor synthesis view must match, not the simulator envelope.
    bind(original/'ddr516_reference.sv');bind(board/'ddr516_reference.sv')
    assert sha(original/'ddr516_reference.sv')==sha(board/'ddr516_reference.sv')
    return dict(status='STATIC_STARTUP_CONTRACT_CHECKED',hardware_access=False,board_release=False,
        clocks_mhz=freq,fvco_mhz=fpll*o,fpll_mhz=fpll,dynamic_output=2,pins=pin_checks,
        rom=dict(words=len(words),changed_sequences=changes,init_unchanged=True,mrs=mrs,
                 vendor_original=False,source_sha256=sha(board/'ddr3_controller.bin')),
        checks=['current IP only','vendor protected synthesis view unchanged','mode-register encoding',
                'clock calculation','GPIO register/clock modes','calibration qualification','host traffic gated until calibration'],
        unresolved=['CLKOUT0 400 MHz exceeds the datasheet 4x25 MHz wording; the official engboard also exceeds 4x50 MHz. Tool legality does not resolve this documentation discrepancy.',
                    'Physical PLL lock and phase movement are not proven by frequency declarations.',
                    'Two-cycle RX retiming is supported by a digital A/B but not yet validated on this board.',
                    'DDR initialization/calibration state inside the protected IP and actual DQ capture remain unobserved.'],
        inputs=records)

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    result=audit(a.board.resolve());result['auditor_sha256']=sha(__file__)
    for path,digest in result['inputs'].items():assert sha(path)==digest
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','pins')},indent=2))

if __name__=='__main__':main()
