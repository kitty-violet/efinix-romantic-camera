"""Four measured SRL replacements; exact surrounding graph and cut accounting."""
import collections,json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from native_local_drivers import CELL,raw_cells,lexical_tokens,render_token,rewrite_expression
from audit_continuous_vectors import expand_graph
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
PARENT=ROOT/'tmp/frame-execution/connected-board-continuous-control1'
OUT=ROOT/'tmp/frame-execution/continuous-srl-register-native1'
TARGETS=['video_processing_pipeline/pipeline/cnn/core/dma/FEATURE_BIT[30].payload',
 'video_processing_pipeline/pipeline/cnn/core/dut/engine/rq/QUEUE[0].BIT[3].storage',
 'video_processing_pipeline/pipeline/cnn/core/dut/engine/rq/QUEUE[0].BIT[4].storage',
 'video_processing_pipeline/pipeline/cnn/core/dut/engine/rq/QUEUE[0].BIT[76].storage']

def replacement(index,pins,initial):
    prefix=f'CONT_SRL_EXPAND_{index}_'; wires=[];blocks=[];spec={}
    def wire(name):
        name=prefix+name;wires.append(name);return name
    def cell(name,kind,p,parameters):
        name=prefix+name;assert name not in spec
        blocks.append('    '+kind+' '+name+' ('+', '.join('.'+k+'('+v+')' for k,v in p.items())+') /* exact SRL expansion */ ;')
        blocks.extend(f"    defparam {name}.{key} = {width}'h{value:x};" for key,(width,value) in parameters.items())
        spec[name]=dict(kind=kind,pins=p,params=parameters)
    def ff(name,d,q,polarity):
        cell(name,'EFX_FF',dict(D=d,CLK=pins['CLK'],CE=pins['CE'],SR="1'b0",Q=q),dict(CLK_POLARITY=(1,1),CE_POLARITY=(1,1),SR_POLARITY=(1,1),SR_SYNC=(1,1),SR_SYNC_PRIORITY=(1,1),SR_VALUE=(1,0),D_POLARITY=(1,polarity)))
    def lut(name,i0,i1,i2,q,mask=0xcaca):
        cell(name,'EFX_LUT4',dict(I0=i0,I1=i1,I2=i2,I3="1'b0",O=q),dict(LUTMASK=(16,mask)))
    state=[wire('data'+str(i)) for i in range(8)]
    for i in range(8):ff('D'+str(i),pins['D'] if i==0 else state[i-1],state[i],0 if i==0 else 1)
    values=state
    if initial==0:
        occupied=[wire('occupied'+str(i)) for i in range(8)];values=[wire('value'+str(i)) for i in range(8)]
        for i in range(8):
            ff('V'+str(i),"1'b1" if i==0 else occupied[i-1],occupied[i],1)
            lut('INIT'+str(i),state[i],occupied[i],"1'b0",values[i],0xbbbb)
    else:assert initial==255
    address=lexical_tokens(pins['A']);assert len(address)==3
    a=[render_token(t) for t in reversed(address)]
    lower=[wire('lower'+str(i)) for i in range(4)];upper=[wire('upper'+str(i)) for i in range(2)]
    for i in range(4):lut('M0_'+str(i),values[2*i+1],values[2*i],a[0],lower[i])
    for i in range(2):lut('M1_'+str(i),lower[2*i+1],lower[2*i],a[1],upper[i])
    lut('OUTPUT',upper[1],upper[0],a[2],pins['Q'])
    return '\n'.join('    wire '+n+';' for n in wires)+'\n'+'\n'.join(blocks)+'\n',spec

def main():
    assert not OUT.exists()
    short=ROOT/'tmp/frame-execution/srl8-register-alternative3';r=json.loads((short/'result.json').read_text())
    assert r['status']=='OFFICIAL_SRL_REGISTER_ALTERNATIVE_KNOWN_ADDRESS_SHORT_PASS'
    assert r['inputs_sha256']==sha(short/'inputs.json') and r['simulation_sha256']==sha(short/'simulate.log')
    for f,h in json.loads((short/'inputs.json').read_text())['evidence'].items():assert sha(f)==h,f
    src=PARENT/'outflow/ti60f225_oob.map.v';text=src.read_text();raw=raw_cells(text);old=expand_graph(text);par=params(text)
    declarations=[];spec={};contracts=[];aliases={};expected=[]
    def alias(term,target):
        value=lexical_tokens(target)[0]
        assert term not in aliases or aliases[term]==value,(term,target,aliases.get(term))
        aliases[term]=value
    for index,name in enumerate(TARGETS):
        kind,pins,_=raw[name];pp=par[name]
        assert kind=='EFX_SRL8' and set(pins)=={'D','CLK','CE','Q','A'}
        assert pp==dict(INIT=(8,pp['INIT'][1]),CLK_POLARITY=(1,1),CE_POLARITY=(1,1)) and pp['INIT'][1] in (0,255)
        assert pins['CLK']=='process_clk_150' and len(lexical_tokens(pins['D']))==1
        code,ss=replacement(index,pins,pp['INIT'][1]);assert not set(ss)&set(old)
        declarations.append(code);spec.update(ss)
        for port in ('CLK','D','CE'):alias(lexical_tokens(pins[port])[0],port)
        for bit,term in enumerate(reversed(lexical_tokens(pins['A']))):alias(term,f'A[{bit}]')
        alias(lexical_tokens(pins['Q'])[0],f'Q[{index}]')
        contracts.append(dict(cell=name,original_pins=pins,original_params=pp,new_cells=list(ss),known_address_required=True,Q7_unconnected=True,additional_ce_loads=sum(x['kind']=='EFX_FF' for x in ss.values())-1))
        block=next(m[0] for m in CELL.finditer(text) if m[2].lstrip('\\')==name)
        dp='\n'.join(m[0] for m in re.finditer(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',text,re.M) if m[1].lstrip('\\')==name)
        expected.append(block+'\n'+dp)
    changed=CELL.sub(lambda m:'' if m[2].lstrip('\\') in TARGETS else m[0],text)
    changed=re.sub(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',lambda m:'' if m[1].lstrip('\\') in TARGETS else m[0],changed,flags=re.M)
    newblock='\n'.join(declarations)
    assert len(re.findall(r'^endmodule\b',changed,re.M))==1
    changed=re.sub(r'^endmodule\b',lambda m:newblock+'\nendmodule',changed,flags=re.M)
    ng=expand_graph(changed);np=params(changed)
    assert set(ng)==(set(old)-set(TARGETS))|set(spec)
    for n in set(old)-set(TARGETS):assert ng[n]==old[n] and np.get(n)==par.get(n),n
    for n,row in spec.items():
        assert ng[n][0]==row['kind'] and ng[n][1]=={p:lexical_tokens(v) for p,v in row['pins'].items()}
        assert np[n]==row['params']
    assert protected_blocks(changed)==protected_blocks(text)
    for pattern in (r'^\s*(?:input|output|inout)\b.*$',r'^\s*assign\b.*$'):
        assert re.findall(pattern,changed,re.M)==re.findall(pattern,text,re.M)
    def mini(block,module):
        def rewrite(m):
            ports=re.sub(r'\.(\w+)\(([^()]*)\)',lambda p:'.'+p[1]+'('+rewrite_expression(p[2],aliases)+')',m[3])
            return m[0][:m.start(3)-m.start()]+ports+m[0][m.end(3)-m.start():]
        body=CELL.sub(rewrite,block)
        return 'module '+module+'(input CLK,CE,D,input [2:0] A,output [3:0] Q);\n'+body+'\nendmodule\n'
    OUT.mkdir();(OUT/'native_source.v').write_text(changed,encoding='utf8')
    (OUT/'candidate_slice.sv').write_text(mini(newblock,'candidate_slice'))
    (OUT/'reference_slice.sv').write_text(mini('\n'.join(expected),'reference_slice'))
    (OUT/'replacement_contracts.json').write_text(json.dumps(contracts,indent=2))
    delta=collections.Counter(row['kind'] for row in spec.values());delta['EFX_SRL8']=-4
    receipt=dict(status='FOUR_SLOW_SRL_EXACT_GRAPH_CANDIDATE_PENDING_ACTUAL_SLICE_TEST',source_sha256=sha(OUT/'native_source.v'),retained_primitives=len(old)-4,removed_cells=TARGETS,added_cells=len(spec),resource_delta=dict(delta),latency_cycles_added=0,clock='process_clk_150',known_address_required=True,Q7_all_unconnected=True,additional_ce_loads={row['cell']:row['additional_ce_loads'] for row in contracts},inputs={str(p):sha(p) for p in [Path(__file__),src,short/'result.json',short/'inputs.json',short/'simulate.log',HERE/'variants/continuous_run/srl8_register_alternative3.sv',OUT/'candidate_slice.sv',OUT/'reference_slice.sv',OUT/'replacement_contracts.json']},hardware_access=False,board_release=False)
    (OUT/'result.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='inputs'}),flush=True)
if __name__=='__main__':main()
