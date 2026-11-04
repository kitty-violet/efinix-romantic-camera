"""Specialize only the ninth product to the proven accepted-A {0,4,8} domain."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,sha
from minimal_feedback_board import audit_delta as audit_parent
from control_command_repair import once
from audit_accepted_a_tail import proof

BASE=ROOT/'tmp/frame-execution/connected-board-minimal-feedback1'
FILES=('assembly/fixed_window_dot8x32.sv','assembly/fixed_ring_weights9.sv',
       'assembly/fixed_production_mac.sv','assembly/fixed_video_compute.sv')

def transform(rel,text):
    if rel==FILES[0]:
        text=once(text,'SHARED_COLUMN_ACTIVATION=0)(','SHARED_COLUMN_ACTIVATION=0, TAIL_SHIFT_048=0)(')
        old='   always @(posedge clk)if(pair_advance[r*4+c/2])product<=$signed(tail_a)*$signed(tail_b);'
        new='''   if(TAIL_SHIFT_048!=0)begin:ACCEPTED_A_TAIL
    // Loader rejects every other ninth-tap coefficient before activation.
    // The two selects are mutually exclusive: 0, 4, 8. This is exact for
    // all signed INT8 operands, including -128; no extra pipeline stage.
    wire [15:0] times4={{6{tail_a[7]}},tail_a,2'b00};
    wire [15:0] times8={{5{tail_a[7]}},tail_a,3'b000};
    always @(posedge clk)if(pair_advance[r*4+c/2])
     product<=({16{tail_b[2]}} & times4) | ({16{tail_b[3]}} & times8);
   end else begin:GENERAL_TAIL
    always @(posedge clk)if(pair_advance[r*4+c/2])product<=$signed(tail_a)*$signed(tail_b);
   end'''
        return once(text,old,new)
    if rel==FILES[1]:
        text=once(text,'parameter integer SKEW_TERMS=0)(','parameter integer SKEW_TERMS=0, TAIL_SHIFT_048=0)(')
        old=' wire good=load_index==count && load_last==last;'
        new=''' wire [7:0] tail_supported;
 for(genvar c=0;c<8;c=c+1)begin:TAIL_GUARD
  // Evaluate the actual incoming byte, not a truncated/cache copy.
  assign tail_supported[c]=load_data[c*8+:8]==8'd0 ||
   load_data[c*8+:8]==8'd4 || load_data[c*8+:8]==8'd8;
 end
 wire good=load_index==count && load_last==last &&
  (TAIL_SHIFT_048==0 || term!=8 || (&tail_supported));'''
        return once(text,old,new)
    if rel==FILES[2]:
        text=once(text,'module fixed_production_mac(','module fixed_production_mac #(parameter integer TAIL_SHIFT_048=0)(')
        text=once(text,'#(.SKEW_TERMS(1)) cache','#(.SKEW_TERMS(1),.TAIL_SHIFT_048(TAIL_SHIFT_048)) cache')
        return once(text,'.SHARED_COLUMN_ACTIVATION(1)) arithmetic',
                         '.SHARED_COLUMN_ACTIVATION(1),.TAIL_SHIFT_048(TAIL_SHIFT_048)) arithmetic')
    if rel==FILES[3]:return once(text,' fixed_production_mac mac(',
                               ' fixed_production_mac #(.TAIL_SHIFT_048(1)) mac(')
    raise ValueError(rel)

def audit_delta(board):
    audit_parent(BASE);domain=proof();meta=json.loads((board/'inputs.json').read_text())
    assert meta['parent']==str(BASE) and meta['parent_inputs_sha256']==sha(BASE/'inputs.json')
    assert meta['accepted_tail_transform_sha256']==sha(__file__)
    assert meta['tail_domain_proof']==json.loads(json.dumps(domain))
    parent=json.loads((BASE/'inputs.json').read_text());expected={Path(r['snapshot']).relative_to(BASE).as_posix() for r in parent['inputs']}
    seen=set();changed=[]
    for r in meta['inputs']:
        f=Path(r['snapshot']);assert f.is_relative_to(board) and sha(f)==r['sha256']
        rel=f.relative_to(board).as_posix();seen.add(rel)
        if rel in FILES:
            assert f.read_text()==transform(rel,(BASE/rel).read_text());changed.append(rel)
        else:assert sha(f)==sha(BASE/rel),rel
    assert seen==expected and set(changed)==set(FILES)
    return dict(status='ACCEPTED_A_TAIL_048_WITH_LOAD_GUARD_DELTA',changed=sorted(changed),
        package_sha256=domain['package_sha256'],allowed_tail_coefficients=[0,4,8],
        unsupported_tail_load_aborts=True,default_generic_modules_preserved=True,
        public_latency_cycles_added=0,coefficient_package_changed=False,topology_changed=False,
        full_video_a_chain_preserved=True,pins_clocks_constraints_unchanged=True,
        hardware_access=False,board_release=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    audit_parent(BASE);root.mkdir();meta=json.loads((BASE/'inputs.json').read_text())
    for row in meta['inputs']:
        src=Path(row['snapshot']);assert src.is_relative_to(BASE) and sha(src)==row['sha256']
        rel=src.relative_to(BASE);dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if rel.as_posix() in FILES:dst.write_text(transform(rel.as_posix(),src.read_text()),encoding='utf8')
        row.update(snapshot=str(dst),sha256=sha(dst))
    meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),
        accepted_tail_transform_sha256=sha(__file__),tail_domain_proof=proof(),
        warning_reference_base=str(BASE),runner_sha256=sha(__file__),hardware_access=False,board_download_allowed=False)
    (root/'runner.py').write_bytes(Path(__file__).read_bytes())
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
    print('ACCEPTED_TAIL_048_BOARD_PREPARED_NOT_RELEASED')

if __name__=='__main__':main()
