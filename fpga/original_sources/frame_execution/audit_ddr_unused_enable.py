"""Explain changed encrypted-IP warnings using unchanged source and mapped sinks.

This is not PHY equivalence or timing qualification. No IP source is edited.
"""
import argparse,collections,json,re
from pathlib import Path
from run_units import ROOT,sha
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output-name',default='ddr_unused_enable_audit.json');a=p.parse_args()
assert Path(a.output_name).name==a.output_name
folders=[a.base.resolve(),a.candidate.resolve()]
evidence=[]
signal='u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/controller_main/r_rd_fifo_wr_en_p'
for folder in folders:
 assert folder.is_relative_to(ROOT/'tmp')
 result=json.loads((folder/'result.json').read_text());net=folder/'outflow/ti60f225_oob.map.v'
 assert sha(net)==result['map_sha256']
 body=net.read_text();refs=re.findall(re.escape('\\'+signal)+r'\s+\[(\d+)(?::(\d+))?\]',body)
 used=sorted(set(bit for hi,lo in refs for bit in range(int(lo or hi),int(hi)+1)))
 assert used==list(range(6,14)),used
 port=re.search(re.escape('.\\'+signal)+r'\s+\((.*?)\),',body,re.S).group(1)
 normalized=re.sub(r'Open_\d+','Open',re.sub(r'\s+',' ',port)).strip()
 opens=re.findall(r'\bOpen_\d+\b',port)
 assert len(opens)==24
 open_counts=collections.Counter(re.findall(r'\bOpen_\d+\b',body))
 # Every apparently unused bit is a truly unloaded net, not an ignored sink.
 for name in opens:
  # Efinity emits these as implicit, single-use nets without declarations.
  assert open_counts[name]==1,name
 evidence.append(dict(folder=str(folder),map_sha256=sha(net),ddr_source_sha256=sha(folder/'ddr516_reference.sv'),used_bits=used,open_bits=list(range(0,6))+list(range(14,32)),port=normalized))
assert evidence[0]['ddr_source_sha256']==evidence[1]['ddr_source_sha256']
assert evidence[0]['port']==evidence[1]['port']
out=folders[1]/a.output_name;assert not out.exists()
data=dict(status='UNCHANGED_UNUSED_DDR_ENABLE_BITS',evidence=evidence,scope='Identical encrypted source and mapped 32-bit export connectivity; bits 24..29 have no sinks in either netlist. Does not prove full PHY equivalence.',board_release=False)
out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(dict(status=data['status'],used_bits=evidence[1]['used_bits'])))
