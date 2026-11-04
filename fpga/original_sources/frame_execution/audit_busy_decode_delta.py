"""Bind the preserved busy decode to its measured production CNN source.

Only a same-value wire alias and synthesis attribute may differ. This is not
physical equivalence or a new whole-board performance measurement.
"""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,sha
p=argparse.ArgumentParser()
p.add_argument('--base',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--frame',type=Path,required=True)
a=p.parse_args()
base=a.base.resolve();candidate=a.candidate.resolve();frame=a.frame.resolve()
assert all(x.is_relative_to(ROOT/'tmp') for x in (base,candidate,frame))
def frozen(folder,board=False):
    meta=json.loads((folder/'inputs.json').read_text())
    result=json.loads((folder/'result.json').read_text())
    assert sha(folder/'inputs.json')==result['inputs_sha256']
    records={}
    selected=None
    if board:
        xml=ET.parse(folder/'ti60f225_oob.xml')
        selected={(folder/n.attrib['name']).resolve() for n in xml.findall(
            './/{http://www.efinixinc.com/enf_proj}design_file')}
        assert selected=={Path(n).resolve() for n in meta['source_files']}
    for row in meta['inputs']:
        f=Path(row['snapshot']);assert sha(f)==row['sha256']
        if selected is None or f.resolve() in selected:
            assert f.name not in records
            records[f.name]=f
    return records
b=frozen(base,True);c=frozen(candidate,True);f=frozen(frame)
bench={'efx_dpram10.v','efx_dsp24.v','efx_dsp48.v','efx_ram10.v',
       'efx_srl8.v','tb_fourway_full.sv'}
common=f.keys()-bench
assert len(common)==44 and common<=b.keys() and common<=c.keys()
delta=''' // Keep the state-only busy decoder separate from the completion cone.
 // COMB4 sharing can otherwise connect a long unused input arc to busy.
 (* syn_keep = "true" *) wire owner_busy_decode=state!=IDLE;
 assign busy=owner_busy_decode;'''
rows=[]
for name in sorted(common):
    before=b[name].read_text();measured=f[name].read_text();after=c[name].read_text()
    def normalize_debug(t):
        return t.replace(' // synthesis translate_off\n','').replace(' // synthesis translate_on\n','')
    assert normalize_debug(before)==normalize_debug(measured),name
    alias_only=name=='row_pair_compute_cluster.sv'
    if alias_only:
        assert after.count(delta)==1
        after=after.replace(delta,' assign busy=state!=IDLE;')
    assert before==after,name
    rows.append(dict(name=name,base_sha256=sha(b[name]),candidate_sha256=sha(c[name]),
                     frame_sha256=sha(f[name]),same_value_alias_only=alias_only))
out=candidate/'busy_alias_equivalence.json';assert not out.exists()
result=dict(status='SAME_VALUE_BUSY_ALIAS_ONLY',files=rows,frame=str(frame),
            frame_result_sha256=sha(frame/'result.json'),base_inputs_sha256=sha(base/'inputs.json'),
            candidate_inputs_sha256=sha(candidate/'inputs.json'),
            physical_equivalence_proven=False,whole_board_performance_proven=False,
            note='43 files unchanged, one direct expression replaced by an identical-value continuous wire. Physical implementation still requires new P&R/STA.')
out.write_text(json.dumps(result,indent=2)+'\n')
print('PASS SAME_VALUE_BUSY_ALIAS_ONLY files=44 unchanged=43 alias=1')
