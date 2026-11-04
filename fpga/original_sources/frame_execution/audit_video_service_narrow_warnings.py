"""Current schema5 warning classification, including exact protected-IP notices."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_synthesis.py')
code=source.read_text()
old="name.removeprefix('protected_inst/')"
assert code.count(old)==1
code=code.replace(old,"name.replace('protected_inst/','')")
needle=" protected_redundant.update("
assert code.count(needle)==1
extra=[
 'dvi_encoder_m0/encg/n1q_m_c[3:1]',
 'dvi_encoder_m0/n323[0]',
 'dvi_encoder_m0/n33[0]',
 'dvi_encoder_m0/ram_b_dat_r[0]_c[7:1]',
 'inst_efx_csi2_rx/u_efx_csi2_rx/csi2_rx_top_inst/n43[0]',
 'inst_efx_csi2_rx/u_efx_csi2_rx/csi2_rx_top_inst/n43[17:14]',
]
prefix='u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/controller_main/'
extra += [prefix+n for n in ('Latency_counter_c_c[15:9]','Latency_counter_c_c[1:0]','n1275_c_c[0]','n1275_c_c[15:2]')]
code=code.replace(needle,' protected_redundant.update('+repr(extra)+')\n'+needle)
exec(compile(code,str(source)+':schema5-current-warning-list','exec'),globals())
