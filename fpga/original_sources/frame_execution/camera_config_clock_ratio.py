"""Correct only the newly exercised PLL4 config/core timing ratio.

Actual clocks stay 5 and 150 MHz with zero phase. The existing conservative
6.666 ns core anchor is retained; config tightens to exactly 30 times it.
No pixel/external-CSI clocks, uncertainties or exceptions are changed.
"""
import xml.etree.ElementTree as ET

def transform(sdc,peri):
    pll=next(x for x in ET.fromstring(peri).iter() if x.tag.endswith('}pll') and x.get('name')=='pll_inst4')
    assert pll.get('pll_def')=='PLL_BR0'
    outs={x.get('name'):x for x in pll if x.tag.endswith('}comp_output_clock')}
    for name,div in (('process_clk_150',4),('CLK_5M',120)):
        assert outs[name].get('out_divider')==str(div)
        assert outs[name].get('phase_setting')=='0' and outs[name].get('is_inverted')=='false'
    assert 'create_clock -period 6.666 -name process_clk_150 [get_ports {process_clk_150}]' in sdc
    old='create_clock -period 200.000 -name CLK_5M [get_ports {CLK_5M}]'
    new='create_clock -period 199.980 -name CLK_5M [get_ports {CLK_5M}]'
    assert sdc.count(old)==1
    return sdc.replace(old,new),dict(clock='CLK_5M',before_ns=200.0,after_ns=199.98,ratio_to_core=30,
        physical_pll_unchanged=True,exceptions_added=0,uncertainties_unchanged=True,pixel_clock_unchanged=True)
