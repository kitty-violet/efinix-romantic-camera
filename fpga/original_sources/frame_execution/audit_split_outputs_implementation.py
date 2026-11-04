"""Apply the unchanged reset/pin implementation audit to the split-output map."""
import argparse,json
from pathlib import Path
from run_units import sha
from split_logic_outputs_board import audit_delta
import audit_control_state_implementation as source

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--board',type=Path,required=True)
    p.add_argument('--physical',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    assert not a.out.exists()
    source.audit_delta=audit_delta
    result=source.audit(a.board.resolve(),a.physical.resolve())
    result['split_output_wrapper_sha256']=sha(__file__)
    a.out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(result['status'])

if __name__=='__main__':main()
