"""Find a bounded replacement inside the existing sensor expectation LUTs.

This is an offline source experiment, not a timing or board release.  It
never adds an override mux to the old result and never edits another domain.
"""
import itertools
import json
import pickle
import copy
import re
import shutil
from pathlib import Path

from run_units import ROOT, HERE, EF, sha
from native_local_drivers import graph as parse_graph, render_token
from prepare_video_service_color_exposure import read_rom

PARENT = ROOT / 'tmp/frame-execution/connected-board-video-service-probe60-input-isp-explicit'
OUT = ROOT / 'tmp/frame-execution/video-sensor74-minimal6'


def emit_candidate(old, original_params, selected, report):
    source=PARENT/'native_source.v'
    original=source.read_text(encoding='utf8')
    body=original
    g=copy.deepcopy(old);ps=copy.deepcopy(original_params)
    edits=[]
    for name,r in selected.items():
        pat=rf'EFX_LUT4\s+{re.escape(name)}\s*\([^;]*;'
        matches=list(re.finditer(pat,body));assert len(matches)==1
        before=matches[0][0];assert parse_graph(before)[name]==('EFX_LUT4',r['old_pins'])
        attribute=re.search(r'/\*.*?\*/',before,re.S)[0]
        attribute,n=re.subn(r"LUTMASK=16'h[0-9a-fA-F]+",f"LUTMASK=16'h{r['mask']:04x}",attribute);assert n==1
        after='EFX_LUT4 '+name+' ('+', '.join('.'+p+'('+render_token(v[0])+')' for p,v in r['pins'].items())+') '+attribute+' ;'
        assert parse_graph(after)[name]==('EFX_LUT4',r['pins'])
        body=body.replace(before,after,1);edits.append((before,after))
        pat=rf'defparam\s+{re.escape(name)}\.LUTMASK\s*=\s*16\x27h([0-9a-fA-F]+)\s*;'
        matches=list(re.finditer(pat,body));assert len(matches)==1
        before=matches[0][0];assert int(matches[0][1],16)==r['old_mask']
        after=f"defparam {name}.LUTMASK = 16'h{r['mask']:04x};"
        body=body.replace(before,after,1);edits.append((before,after))
        g[name]=('EFX_LUT4',r['pins']);ps[name]['LUTMASK']=(16,r['mask'])
    words,blocks=read_rom(body)
    changes={116:(0x3e01,0x46),117:(0x3e02,0),162:(0x3e09,0x34)}
    lo,hi,block,width,blob=next(b for b in blocks if b[3]==20)
    updated=blob;newwords=list(words)
    for i,(reg,val) in changes.items():
        assert words[i]>>9==reg
        newwords[i]=reg<<9|val<<1
        updated=(updated&~(0xfffff<<(i*20)))|((newwords[i]&0xfffff)<<(i*20))
    changed=block;ram='camera_checked/config_rom/ROM__D$2'
    for i in range(40):
        a=(blob>>(256*i))&((1<<256)-1);b=(updated>>(256*i))&((1<<256)-1)
        if a==b:continue
        param='INIT_'+format(i,'X');ps[ram][param]=(256,b)
        changed,n=re.subn('('+param+r"\s*=\s*256'b)"+f'{a:0256b}',lambda m:m[1]+f'{b:0256b}',changed);assert n==2
    body=body[:lo]+changed+body[hi:]
    restored=body[:lo]+block+body[lo+len(changed):]
    for before,after in reversed(edits):
        assert restored.count(after)==1
        restored=restored.replace(after,before,1)
    assert restored==original,'Unlisted source edit'
    assert read_rom(body)[0]==newwords
    assert g.keys()==old.keys() and {n for n in g if g[n]!=old[n]}=={n for n in selected if selected[n]['pins']!=selected[n]['old_pins']}
    assert {n for n in ps if ps[n]!=original_params[n]}==set(selected)|{ram}
    (OUT/'native_source.v').write_text(body,encoding='utf8')
    # Structural fanout remains inside expected-data FF inputs. Bit 5 shares
    # one index decode but its full D function is proved unchanged above.
    outputs={'EFX_LUT4':['O'],'EFX_FF':['Q'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA']}
    consumers={}
    for n,(k,pins) in g.items():
        for p,ts in pins.items():
            if p not in outputs.get(k,[]):
                for t in ts:consumers.setdefault(t,[]).append((n,p))
    ends=set();todo=[g[n][1]['O'][0] for n in selected];seen=set()
    while todo:
        t=todo.pop()
        if t in seen:continue
        seen.add(t)
        for n,p in consumers.get(t,[]):
            if g[n][0]=='EFX_LUT4':todo+=g[n][1]['O']
            else:ends.add((n,p))
    assert ends=={(f'camera_config_diagnostic[{16+b}]~FF','D') for b in range(1,6)},ends
    drivers={t:n for n,(k,pins) in g.items() for p in outputs.get(k,[]) for t in pins.get(p,[])}
    boundary={'CLK_5M':'clk','w_i2c_arstn':'reset_n','io_cam_scl_IN':'scl_i','io_cam_sda_IN':'sda_i'}
    roots={p:('net',n,None) for p,n in [('scl_oen','camera_new_scl_oen'),('sda_oen','camera_new_sda_oen'),('config_ok','camera_config_ok'),('config_failed','camera_config_failed')]}
    roots.update({f'diagnostic[{i}]':('net','camera_config_diagnostic',i) for i in range(64) if i!=59})
    stack=list(roots.values());cells=set();outside=set()
    while stack:
        t=stack.pop()
        if t[0]=='literal':continue
        if t[1] in boundary:outside.add(t);continue
        assert t in drivers,('undriven',t)
        n=drivers[t]
        if n in cells:continue
        cells.add(n);k,pins=g[n];assert k in outputs
        stack.extend(t for p,ts in pins.items() if p not in outputs[k] for t in ts)
    nets={t:f'n{i}' for i,t in enumerate(sorted({t for n in cells for ts in g[n][1].values() for t in ts if t[0]!='literal'}))}
    lines=['module camera_checked_config #(parameter PRESCALE=32)(input clk,reset_n,scl_i,sda_i,output scl_o,scl_oen,sda_o,sda_oen,config_ok,config_failed,output [63:0] diagnostic);','wire '+','.join(nets.values())+';']
    for t in sorted(outside):lines.append(f'assign {nets[t]}={boundary[t[1]]};')
    for p,t in roots.items():lines.append(f'assign {p}={nets[t]};')
    lines+=['assign diagnostic[59]=0; assign scl_o=0; assign sda_o=0;']
    def expr(ts):
        v=[t[1] if t[0]=='literal' else nets[t] for t in ts]
        return v[0] if len(v)==1 else '{'+','.join(v)+'}' if v else ''
    for n in sorted(cells):
        k,pins=g[n];lines.append(k+' \\'+n+' ('+','.join('.'+p+'('+expr(ts)+')' for p,ts in pins.items())+');')
        for p,v in ps[n].items():lines.append('defparam \\'+n+' .'+p+'='+(str(v) if not isinstance(v,tuple) else f"{v[0]}'h{v[1]:x}")+';')
    lines.append('endmodule')
    (OUT/'mapped_camera.v').write_text('\n'.join(lines),encoding='utf8')
    (OUT/'expected.hex').write_text(''.join(f'60{w>>9:04x}{(w>>1)&255:02x}\n' for w in newwords))
    # Reference and expected write list are generated from the SAME final ROM.
    final={w>>9:(w>>1)&255 for w in newwords};gain=final[0x3e09]
    reference=(HERE/'camera_checked_config.sv').read_text(encoding='utf8')
    assert reference.count("3e092aff")==1
    (OUT/'camera_checked_config_reference.sv').write_text(reference.replace('3e092aff',f'3e09{gain:02x}ff'),encoding='utf8')
    tb=(HERE/'tb_camera_checked_config.sv').read_text()
    tb=tb.replace("if(mode==3 && address==16'h3107)","if(mode==10 && address==16'h3e09) send_byte=8'h2a;\n          if(mode==3 && address==16'h3107)")
    tb=tb.replace('mode>=7','(mode>=7 && mode<=9)')
    tb=tb.replace('if(mode==5 && diagnostic',f'if(mode==10 && (diagnostic[59:56]!=5 || diagnostic[47:32]!=16\'h3e09 || diagnostic[23:16]!=8\'h{gain:02x})) $fatal(1,"GAIN_MISMATCH_NOT_REJECTED");\n      if(mode==5 && diagnostic')
    (OUT/'tb.sv').write_text(tb)
    kinds=sorted({g[n][0] for n in cells})
    for k in kinds:shutil.copy2(EF/'sim_models/verilog'/(k.lower()+'.v'),OUT/(k.lower()+'.v'))
    proof=dict(status='SENSOR74_MINIMAL_CANDIDATE_NOT_SYNTHESIZED',gain_register=gain,exposure_half_lines=1120,
        source_sha256=sha(OUT/'native_source.v'),original_synthesis_input_sha256=sha(source),script_sha256=sha(__file__),
        new_luts=0,new_flip_flops=0,new_clocks=0,changed_ram_address_pins=0,changed_ff_d_pins=0,
        all_unlisted_source_byte_equal=True,verification_still_checks_all_11_registers=True,
        altered_lut_outputs_reach_only_expected_d=sorted(ends),cells=len(cells),kinds=kinds,
        files={str(f):sha(f) for f in OUT.iterdir() if f.is_file()},hardware_access=False,board_release=False)
    (OUT/'inputs.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(proof['status'],'cells',len(cells),flush=True)


def main():
    assert not OUT.exists()
    OUT.mkdir()
    key, src, graph, sp, params = pickle.loads((PARENT/'retained-audit-parse.pkl').read_bytes())
    assert key[1] == sha(PARENT/'outflow/ti60f225_oob.map.v')
    drivers = {t: n for n, (k, p) in graph.items() for t in p.get('O', []) if k == 'EFX_LUT4'}
    # Bit-parallel exhaustive domain: phase, all index values, ROM bit, old FF.
    count = 16*256*4
    full = (1 << count)-1
    leaves = {}
    for b in range(4):
        leaves[('net','camera_config_diagnostic',60+b)] = sum(1 << s for s in range(count) if ((s//1024)>>b)&1)
    for b in range(8):
        leaves[('net','camera_config_diagnostic',48+b)] = sum(1 << s for s in range(count) if (((s//4)%256)>>b)&1)
    rom = sum(1 << s for s in range(count) if s&2)
    held = sum(1 << s for s in range(count) if s&1)
    toggle = sum(1 << s for s in range(count) if s//1024 == 4 and (s//4)%256 == 10)

    def evaluate(token, values, replacements=None, forced=None):
        if token[0] == 'literal':
            assert token[1] in ("1'b0", "1'b1"), token
            return full if token[1] == "1'b1" else 0
        if token in values:
            return values[token]
        name = drivers[token]
        if forced and name in forced:
            return forced[name]
        pins, mask = (replacements or {}).get(name, (graph[name][1],params[name]['LUTMASK'][1]))
        ins = [evaluate(pins.get('I'+str(i), [('literal',"1'b0",None)])[0], values, replacements, forced) for i in range(4)]
        result = 0
        for p in range(16):
            if (mask >> p)&1:
                part = full
                for i, v in enumerate(ins):
                    part &= v if (p>>i)&1 else full^v
                result |= part
        values[token] = result
        return result

    def solve(name, pins, output, base, desired):
        zero = evaluate(output, dict(base), forced={name:0})
        one = evaluate(output, dict(base), forced={name:full})
        if ((zero ^ desired) & (one ^ desired)):
            return None
        need_one, need_zero = zero ^ desired, one ^ desired
        ins = [evaluate(pins.get('I'+str(i), [('literal',"1'b0",None)])[0],dict(base)) for i in range(4)]
        mask = params[name]['LUTMASK'][1]
        for p in range(16):
            group=full
            for i,v in enumerate(ins):
                group &= v if (p>>i)&1 else full^v
            yes, no = bool(group&need_one), bool(group&need_zero)
            if yes and no:
                return None
            if yes: mask |= 1<<p
            if no: mask &= ~(1<<p)
        actual = evaluate(output,dict(base),{name:(pins,mask)})
        assert actual == desired
        return mask

    selected = {}
    failed = []
    for bit in range(1,5):
        base = dict(leaves)
        base[('net','camera_checked/rom_data',bit+1)]=rom
        base[('net','camera_config_diagnostic',16+bit)]=held
        output=('net','camera_checked/n229',bit)
        old = evaluate(output,dict(base))
        desired = old ^ toggle
        # Only LUTs private to each affected expected-data bit.
        candidates={1:range(241551,241554),2:range(241554,241558),3:range(241558,241561),4:range(241561,241564)}[bit]
        found=[]
        for i in candidates:
            name='LUT__'+str(i)
            mask=solve(name,graph[name][1],output,base,desired)
            if mask is not None:
                found.append((0,(mask^params[name]['LUTMASK'][1]).bit_count(),name,graph[name][1],mask))
        fixed = {}
        if not found and bit in (1,4):
            if bit == 4:
                # Move the old final index-bit AND into the existing decode.
                # The final LUT then only combines decoded VERIFY with ID/ROM/hold.
                name='LUT__241563'
                fixed[name]=(graph[name][1],params[name]['LUTMASK'][1])
                params[name]['LUTMASK']=(16,0xcfcf)
            # Reuse existing pure-index predecodes, rather than add a cell.
            pure={('net','camera_config_diagnostic',48+b): leaves[('net','camera_config_diagnostic',48+b)] for b in range(8)}
            for token in drivers:
                try: evaluate(token,pure)
                except KeyError: pass
            funcs={}
            for token,vector in pure.items():
                if token[0]!='net':continue
                # A shared index function must remain unchanged by the other
                # selected expected-bit replacements (combined proof follows).
                changes={n:(r['pins'],r['mask']) for n,r in selected.items()}
                try:
                    if evaluate(token,dict(leaves),changes)!=vector:continue
                except KeyError:continue
                # Preserve original all-index semantics, qualifier remains I3.
                funcs.setdefault(vector,token)
            name='LUT__241551' if bit==1 else 'LUT__241561';original=graph[name][1]
            current=[original['I'+str(i)][0] for i in range(3)]
            ordered=sorted(funcs.values(),key=lambda t:(t not in current,str(t)))
            for cost in (1,2,3):
                for chosen in itertools.combinations(ordered,3):
                    if len(set(chosen)-set(current))!=cost:continue
                    pins={**original,**{'I'+str(i):[t] for i,t in enumerate(chosen)}}
                    mask=solve(name,pins,output,base,desired)
                    if mask is not None:
                        found.append((cost,(mask^params[name]['LUTMASK'][1]).bit_count(),name,pins,mask))
                if found:break
            print('index functions',len(funcs),'bit',bit,'candidates',len(found),flush=True)
        if not found:
            failed.append(bit)
        else:
            found.sort(key=lambda x:(x[0],x[1],x[2],str(x[3])))
            cost, changes, name, pins, mask=found[0]
            selected[name]=dict(bit=bit,pins=pins,old_pins=graph[name][1],mask=mask,old_mask=params[name]['LUTMASK'][1],reused_input_functions=cost)
            for fn,(fp,fm) in fixed.items():
                selected[fn]=dict(bit=bit,pins=fp,old_pins=fp,mask=params[fn]['LUTMASK'][1],old_mask=fm,reused_input_functions=0)
            print('BIT',bit,'CELL',name,'MASK',hex(mask),'input changes',cost,flush=True)
        for fn,(fp,fm) in fixed.items():
            params[fn]['LUTMASK']=(16,fm)
    report=dict(status='MINIMAL_EXPECTATION_REPLACEMENT_FOUND' if not failed else 'MINIMAL_REPLACEMENT_INCOMPLETE',selected=selected,unresolved_bits=failed,
                exhaustive_cases_per_bit=count,added_cells=0,removed_cells=0,hardware_access=False,board_release=False,
                source_map_sha256=key[1],script_sha256=sha(__file__))
    (OUT/'search.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],failed,flush=True)
    if not failed:
        changes={n:(r['pins'],r['mask']) for n,r in selected.items()}
        for bit in range(8):
            base=dict(leaves)
            base[('net','camera_checked/rom_data',bit+1)]=rom
            base[('net','camera_config_diagnostic',16+bit)]=held
            output=('net','camera_checked/n229',bit)
            expected=evaluate(output,dict(base))^(toggle if bit in range(1,5) else 0)
            assert evaluate(output,dict(base),changes)==expected,('combined_delta',bit)
        report['combined_exhaustive_cases']=count*8
        (OUT/'search.json').write_text(json.dumps(report,indent=2)+'\n')
        emit_candidate(graph,params,selected,report)


if __name__ == '__main__':main()
