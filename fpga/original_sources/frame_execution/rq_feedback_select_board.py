"""Use the registered RQ job kind for operands; keep every acceptance guard."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,sha
from control_command_repair import once
from rq_cached_head_board import audit_delta as audit_parent

BASE=ROOT/'tmp/frame-execution/connected-board-cached-head1'
FILES=('assembly/fixed_rq32_mixed.sv','assembly/fixed_skip_vectors4.sv','assembly/fixed_video_compute.sv')

def transform(rel,text):
    if rel==FILES[0]:
        text=once(text,' input wire [26:0] in_x,input wire [287:0] feedback_value,',
            ' input wire [26:0] in_x,input wire [287:0] feedback_value,\n output wire feedback_post_selected,')
        text=once(text,' wire [1:0] chosen=selected_kind;',
            ' wire [1:0] chosen=selected_kind;\n'
            ' // Operand steering comes from the already registered job. Late\n'
            ' // cancel/valid/credit only qualify issue, never the arithmetic data.\n'
            " assign feedback_post_selected=chosen==2'd2;")
    elif rel==FILES[1]:
        text=once(text,'BASE_WIDTH=10,OFFSET_PREAPPLIED=0)',
            'BASE_WIDTH=10,OFFSET_PREAPPLIED=0,REGISTERED_FEEDBACK_SELECT=0)')
        text=once(text,' output wire [287:0] feedback_value,',
            ' input wire feedback_post_selected,output wire [287:0] feedback_value,')
        start=text.index(' // Keep the feedback selector physically independent')
        end=text.index(' wire feedback_select;',start)
        text=text[:start]+(' // Shared production RQ reserves the operation one cycle before issue.\n'
            ' // Use that reservation for data only. All public ready/valid,\n'
            ' // cancel, error and credit guards above remain unchanged.\n'
            ' wire feedback_is_post=REGISTERED_FEEDBACK_SELECT!=0 ? feedback_post_selected:post_ready;\n')+text[end:]
        text=once(text,'.I0(main_scaled),.I1(post_ready)', '.I0(main_scaled),.I1(feedback_is_post)')
        text=once(text,'wire signed [7:0] a=post_ready ?', 'wire signed [7:0] a=feedback_is_post ?')
    elif rel==FILES[2]:
        text=once(text,' wire [287:0] feedback_value;',
            ' wire [287:0] feedback_value;wire feedback_post_selected;')
        text=once(text,'rq(.feedback_value(feedback_value),',
            'rq(.feedback_post_selected(feedback_post_selected),.feedback_value(feedback_value),')
        text=once(text,'.OFFSET_PREAPPLIED(SKIP_OFFSET_PREAPPLIED)) skip(.feedback_value(feedback_value),',
            '.OFFSET_PREAPPLIED(SKIP_OFFSET_PREAPPLIED),.REGISTERED_FEEDBACK_SELECT(1)) skip(.feedback_post_selected(feedback_post_selected),.feedback_value(feedback_value),')
        # If a feedback operation is accepted, reserved kind and accepted kind
        # must agree. Numerical tests also compare all 32 accepted operands.
        text=once(text,' wire [5:0] wv,wr,ws,wsr,wsw,sv,sr,ss,ssr,ssw;',
            ' // synthesis translate_off\n'
            ' always @(posedge clk)if(!backend_reset && !local_flush)begin\n'
            '  if(ir[1] || ir[2])begin\n'
            '   if(feedback_post_selected!==ir[2])$fatal(1,"FEEDBACK_KIND_GUARD_MISMATCH");\n'
            '   for(integer q=0;q<32;q=q+1)begin\n'
            '    if(ir[2] && feedback_value[q*9+:9]!==post_value[q*9+:9])$fatal(1,"FEEDBACK_POST_OPERAND_MISMATCH");\n'
            '    if(ir[1] && feedback_value[q*9+:9]!=={align_value[q*8+7],align_value[q*8+:8]})$fatal(1,"FEEDBACK_ALIGN_OPERAND_MISMATCH");\n'
            '   end\n'
            '  end\n'
            ' end\n'
            ' // synthesis translate_on\n'
            ' wire [5:0] wv,wr,ws,wsr,wsw,sv,sr,ss,ssr,ssw;')
    return text

def audit_delta(board):
    audit_parent(BASE);meta=json.loads((board/'inputs.json').read_text())
    assert meta['parent']==str(BASE) and meta['parent_inputs_sha256']==sha(BASE/'inputs.json')
    assert meta['feedback_select_transform_sha256']==sha(__file__)
    expected={Path(r['snapshot']).relative_to(BASE).as_posix() for r in json.loads((BASE/'inputs.json').read_text())['inputs']}
    actual=set();changed=[]
    for row in meta['inputs']:
        path=Path(row['snapshot']);assert path.is_relative_to(board) and sha(path)==row['sha256']
        rel=path.relative_to(board).as_posix();actual.add(rel)
        if rel in FILES:
            assert path.read_text()==transform(rel,(BASE/rel).read_text());changed.append(rel)
        else:assert sha(path)==sha(BASE/rel),rel
    assert actual==expected and set(changed)==set(FILES)
    return dict(status='RQ_REGISTERED_KIND_OPERAND_SELECTION_ONLY_DELTA',changed=sorted(changed),
        parent_cached_heads=True,acceptance_guards_unchanged=True,new_pipeline_latency=0,
        full_video_a_chain_preserved=True,pins_clocks_constraints_unchanged=True,
        hardware_access=False,board_release=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    audit_parent(BASE);root.mkdir();meta=json.loads((BASE/'inputs.json').read_text())
    for row in meta['inputs']:
        src=Path(row['snapshot']);assert sha(src)==row['sha256']
        rel=src.relative_to(BASE);dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        if rel.as_posix() in FILES:dst.write_text(transform(rel.as_posix(),src.read_text()),encoding='utf8')
        row.update(snapshot=str(dst),sha256=sha(dst))
    meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),
        feedback_select_transform_sha256=sha(__file__),runner_sha256=sha(__file__),
        hardware_access=False,board_download_allowed=False)
    (root/'runner.py').write_bytes(Path(__file__).read_bytes())
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
    print('RQ_REGISTERED_KIND_BOARD_PREPARED_NOT_RELEASED')

if __name__=='__main__':main()
