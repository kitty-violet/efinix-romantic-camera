"""Use existing bounded, identical checks for the minimal original-queue delta."""
import sys,json
from pathlib import Path
from minimal_feedback_board import audit_delta

def bind():
    import feedback_select_flow as f
    f.audit_delta=audit_delta
    return f,f.bind()

def main():
    mode=sys.argv[1];f,old=bind()
    if mode=='selector-audit':
        import audit_feedback_selection_map as check
        check.audit_delta=audit_delta
        board=Path(sys.argv[2]).resolve();out=board/'feedback_selection_map_audit.json'
        assert not out.exists();result=check.audit(board)
        out.write_text(json.dumps(result,indent=2));print(result['status']);return
    assert mode!='head-audit','This version deliberately has no cached queue heads'
    f.main()
if __name__=='__main__':main()
