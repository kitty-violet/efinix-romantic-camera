"""Reject missing/ambiguous Gray FFs and relaxed skew before any CAD launch."""
import json
from run_units import ROOT,sha
from rebind_native_placement import SOURCE,translate

def main():
    out=ROOT/'tmp/board-test/native-rebind-preflight1';assert not out.exists();out.mkdir()
    sdc=(SOURCE/'ti60f225_oob.sdc').read_text()
    net=(SOURCE/'outflow/ti60f225_oob.map.v').read_text()
    placed=(SOURCE/'outflow/ti60f225_oob.out.placeloc').read_text()
    good,mapping=translate(sdc,net,placed)
    old,target=next((a,b) for a,b in mapping.items() if a!=b)
    name=target.rsplit('|',1)[0]
    tests={
        'missing_ff':(sdc,'\n'.join(line for line in placed.splitlines() if not line.startswith(name+'\t'))),
        'ambiguous_ff':(sdc,placed+'\nprotected_inst/'+name+'\t1\t1\t2\n'),
        'relaxed_skew':(sdc.replace('] 2.500','] 2.501',1),placed)
    }
    results=[]
    for case,(text,locs) in tests.items():
        try:translate(text,net,locs)
        except AssertionError as exc:results.append(dict(case=case,rejected=True,reason=str(exc)))
        else:raise AssertionError('Invalid binding accepted: '+case)
    record=dict(status='EXACT_BINDING_AND_THREE_NEGATIVE_CASES_PASS',cases=results,
        positive_pins=len(mapping),renamed_pins=sum(a!=b for a,b in mapping.items()),
        checker_sha256=sha(__file__),adapter_sha256=sha(__file__.replace('test_native_rebind_policy.py','rebind_native_placement.py')),
        hardware_access=False,cad_invoked=False)
    (out/'result.json').write_text(json.dumps(record,indent=2));print(record['status'],flush=True)

if __name__=='__main__':main()
