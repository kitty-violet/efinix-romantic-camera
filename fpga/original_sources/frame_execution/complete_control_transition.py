"""Local implementation alternative for the entire fixed controller D cone.

The selected board fails in silicon while its Boolean and STA checks pass.
This isolates shared next-state implementation, not a proven silicon root cause.
"""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from accepted_tail_board import audit_delta as parent_audit
BASE=ROOT/'tmp/frame-execution/connected-board-accepted-tail2'
REL='assembly/EfxSapphireSoc.v'

def circuit():
    lines=['module control_complete_next(',
      ' input wire [9:0] matched,',
      ' input wire host_valid,boot_valid,decode_bad,mailbox_hit,apb_hit,writing,',
      ' input wire pready,aw_done,w_done,bvalid,arready,rvalid,rlast,',
      ' output wire [9:0] next_state);']
    names=[]
    def lut(name,args,fn,out=None):
        assert len(args)<=4
        if out is None:
            out=name;lines.append(' (* syn_keep="true" *) wire '+name+';')
        mask=sum(1<<v for v in range(16) if fn(*[(v>>i)&1 for i in range(len(args))]))
        pins=args+["1'b0"]*(4-len(args))
        lines.append(f" EFX_LUT4 #(.LUTMASK(16'h{mask:04x})) n_{name}("+
                     ','.join(f'.I{i}({x})' for i,x in enumerate(pins))+f',.O({out}));')
        names.append('n_'+name)
        return out
    s=[f'matched[{i}]' for i in range(10)]
    cmd=lut('command',['host_valid','boot_valid'],lambda a,b:a or b)
    anylo=lut('anylo',s[:4],lambda *x:any(x))
    anymi=lut('anymi',s[4:8],lambda *x:any(x))
    valid=lut('legal',[anylo,anymi,*s[8:]],lambda *x:any(x))
    decode=lut('decode_ok',[s[1],'decode_bad'],lambda a,b:a and not b)
    ddr=lut('ddr',[decode,'mailbox_hit','apb_hit'],lambda a,b,c:a and not b and not c)
    wd=lut('write_done',['aw_done','w_done'],lambda a,b:a and b)
    rd=lut('read_done',['rvalid','rlast'],lambda a,b:a and b)
    lut('idle',[s[0],cmd,s[9],valid],lambda a,b,c,d:(a and not b) or c or not d,'next_state[0]')
    lut('decode',[s[0],cmd],lambda a,b:a and b,'next_state[1]')
    lut('mail',[decode,'mailbox_hit','writing'],lambda a,b,c:a and b and not c,'next_state[2]')
    lut('setup',[decode,'mailbox_hit','apb_hit'],lambda a,b,c:a and not b and c,'next_state[3]')
    lut('access',[s[3],s[4],'pready'],lambda a,b,c:a or (b and not c),'next_state[4]')
    # Both functions use all four inputs. Separate two-input start_read /
    # start_write LUTs were packed into one COMB4 even with syn_keep wires.
    lut('write',[ddr,'writing',s[5],wd],lambda a,b,c,d:(a and b) or (c and not d),'next_state[5]')
    finishw=lut('finish_write',[s[5],wd],lambda a,b:a and b)
    lut('b',[finishw,s[6],'bvalid'],lambda a,b,c:a or (b and not c),'next_state[6]')
    lut('ar',[ddr,'writing',s[7],'arready'],lambda a,b,c,d:(a and not b) or (c and not d),'next_state[7]')
    addr=lut('address_done',[s[7],'arready'],lambda a,b:a and b)
    lut('r',[addr,s[8],rd],lambda a,b,c:a or (b and not c),'next_state[8]')
    drep=lut('decode_reply',[s[1],'decode_bad','mailbox_hit','writing'],lambda a,b,c,d:a and (b or (c and d)))
    prep=lut('apb_reply',[s[4],'pready'],lambda a,b:a and b)
    brep=lut('b_reply',[s[6],'bvalid'],lambda a,b:a and b)
    rrep=lut('r_reply',[s[8],rd],lambda a,b:a and b)
    group=lut('reply_group',[drep,s[2],prep,brep],lambda *x:any(x))
    lut('reply',[group,rrep],lambda a,b:a or b,'next_state[9]')
    lines.append('endmodule\n')
    return '\n'.join(lines),names

def transform(text):
    declaration=' (* syn_keep="true" *) wire [9:0] ctrl_state_match;'
    assert text.count(declaration)==1
    text=text.replace(declaration,declaration+'\n (* syn_keep="true" *) wire [9:0] ctrl_next_state;')
    anchor=' assign axiA_rready=!reset && ctrl_state_match[8];'
    assert text.count(anchor)==1
    instance='''
 control_complete_next ctrl_next(.matched(ctrl_state_match),
  .host_valid(host_valid),.boot_valid(boot_valid),.decode_bad(decode_bad),
  .mailbox_hit(mailbox_hit),.apb_hit(apb_hit),.writing(writing),
  .pready(io_apbSlave_0_PREADY),.aw_done(aw_sent || axiA_awready),
  .w_done(w_sent || axiA_wready),.bvalid(axiA_bvalid),.arready(axiA_arready),
  .rvalid(axiA_rvalid),.rlast(axiA_rlast),.next_state(ctrl_next_state));'''
    text=text.replace(anchor,anchor+instance)
    begin=text.index('   case(1\'b1)',text.index(' // Fixed one-hot control;'))
    end=text.index('   endcase',begin)+len('   endcase')
    case=text[begin:end]
    changed,n=re.subn(r'state_oh<=[^;]+;', ';',case)
    assert n==14,n
    # Keep every non-state side effect and conditional verbatim. Empty
    # conditional branches are intentional: the next-state LUTs own that part.
    text=text[:begin]+'   state_oh<=ctrl_next_state;\n'+changed+text[end:]
    assert len(re.findall(r'state_oh<=',text))==2 # reset + the only clocked D
    return text+'\n'+circuit()[0]

def audit_delta(board):
    parent_audit(BASE)
    meta=json.loads((board/'inputs.json').read_text())
    assert meta['parent']==str(BASE) and meta['parent_inputs_sha256']==sha(BASE/'inputs.json')
    assert meta['complete_transition_transform_sha256']==sha(__file__)
    parent=json.loads((BASE/'inputs.json').read_text())
    expected={Path(r['snapshot']).relative_to(BASE).as_posix() for r in parent['inputs']}
    seen=set();changed=[]
    for row in meta['inputs']:
        f=Path(row['snapshot']);assert f.is_relative_to(board) and sha(f)==row['sha256']
        rel=f.relative_to(board).as_posix();seen.add(rel)
        if rel==REL:assert f.read_text()==transform((BASE/rel).read_text());changed.append(rel)
        else:assert sha(f)==sha(BASE/rel),rel
    assert seen==expected and changed==[REL]
    return dict(status='COMPLETE_CONTROL_TRANSITION_ONLY_DELTA',changed=changed,
       next_state_luts=len(circuit()[1]),protocol_latency_changed=False,
       ddr_changed=False,full_video_a_chain_preserved=True,pins_clocks_constraints_unchanged=True,
       board_release=False,hardware_access=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    parent_audit(BASE);candidate=transform((BASE/REL).read_text())
    root.mkdir();meta=json.loads((BASE/'inputs.json').read_text())
    for row in meta['inputs']:
        src=Path(row['snapshot']);assert src.is_relative_to(BASE) and sha(src)==row['sha256']
        rel=src.relative_to(BASE);dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if rel.as_posix()==REL:dst.write_text(candidate,encoding='utf8')
        row.update(snapshot=str(dst),sha256=sha(dst))
    meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),
      complete_transition_transform_sha256=sha(__file__),warning_reference_base=str(BASE),
      hardware_access=False,board_download_allowed=False)
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
    print('COMPLETE_TRANSITION_BOARD_PREPARED_NOT_RELEASED')
if __name__=='__main__':main()
