"""Current ISP source delta plus retained-map proof with explicit mapper folds."""
import argparse,collections,json,pickle,re,time
from pathlib import Path
from run_units import ROOT,HERE,sha
from continuous_control_native import net
from native_local_drivers import render_token
from probe_full_native_split_map import lut_function
from complete_native_logic_split import protected_blocks


def mapped_folds(a,b,pa,pb,source):
    missing=set(a)-set(b)
    assert missing=={'cnn_run_seen~FF','LUT__133287','LUT__140273'},missing
    def fn(g,p,n):
        return lut_function({k:render_token(v[0]) for k,v in g[n][1].items() if v},p[n]['LUTMASK'][1])
    # Exact duplicate decoder, then an identity buffer. No arbitrary missing
    # cells are accepted and every remaining consumer is still compared.
    clone='LUT__133283_LUT__140223_splitO'
    assert a[clone]==b[clone] and pa[clone]==pb[clone]
    assert fn(a,pa,'LUT__133287')[1:]==fn(a,pa,clone)[1:]
    identity=fn(a,pa,'LUT__140273')
    assert identity[1:]==((render_token(net('n63588_2')),),(0,1)),identity
    assert len(re.findall(r'(?<![\w/])cnn_run_seen(?![\w~])',source))==2
    assert a['cnn_run_seen~FF'][1]['Q']==[net('cnn_run_seen')]
    assert not any(net('cnn_run_seen') in vs for n,(_,ps) in a.items() if n!='cnn_run_seen~FF' for vs in ps.values())
    aliases={net('n63592_2'):net('n63588_2'),net('n66866_2'):net('n63588_2')}
    for n in missing:del a[n]
    for _,pins in a.values():
        for p,ts in pins.items():pins[p]=[aliases.get(t,t) for t in ts]


def source_delta(board):
    meta=json.loads((board/'inputs.json').read_text());native=Path(meta['native'])
    proof=json.loads((native/'input-isp-edit.json').read_text());parent=Path(proof['parent'])
    key,a,b,pa,pb=pickle.loads((board/'retained-audit-parse.pkl').read_bytes())
    assert key[:2]==[sha(board/'native_source.v'),sha(board/'outflow/ti60f225_oob.map.v')]
    h,old,op=pickle.loads((parent/'color-source-parse.pkl').read_bytes())
    assert h==proof['parent_sha256']==sha(parent/'native_source.v')
    assert proof['candidate_sha256']==key[0] and old.keys()==a.keys() and op==pa
    count=0;categories=collections.Counter()
    for n,(kind,pins) in old.items():
        nk,np=a[n];assert nk==kind and pins.keys()==np.keys(),n
        changed=False
        for p,vs in pins.items():
            assert len(vs)==len(np[p])
            for x,y in zip(vs,np[p]):
                if x==y:continue
                changed=True
                if x[1]=='raw_frontend/consumer/rd':
                    assert n.startswith('raw_frontend/consumer/debayer/legacy_debayer/inst_rgb_gain/')
                    assert y==net(x[1],x[2]^8);category='raw_pair_swap'
                elif x[1] in ('pair_pixel_rgb','pair_pixel_frame_id'):
                    assert n.startswith('video_processing_pipeline/downscaler/')
                    assert y==net('service_isp_rgb' if x[1]=='pair_pixel_rgb' else 'service_isp_id',x[2]);category='pixel_id_delay'
                elif x==net('~ceg_net7945'):
                    assert n.startswith('video_processing_pipeline/downscaler/') or n in ('LUT__153494','LUT__153498','LUT__153504','LUT__153506')
                    assert y==net('service_isp_fire');category='downscaler_fire'
                elif n in ('LUT__153493','LUT__153496'):
                    assert p=='I3' and x==net('n74389_2') and y==net('service_isp_ready');category='serializer_ready'
                else:
                    assert n=='LUT__153497'
                    assert (p,x,y) in [('I0',net('raw_frontend/consumer/serializer/fifo_data',66),net('service_isp_sof')),('I1',net('raw_frontend/consumer/serializer/pair_fifo/dequeue'),net('service_isp_fire'))]
                    category='frame_start_delay'
                categories[category]+=1
        count+=changed
    assert count==179 and len(a)==54243
    assert protected_blocks((parent/'native_source.v').read_text())==protected_blocks((board/'native_source.v').read_text())
    for f in parent.glob('*.sv'):assert sha(f)==sha(native/f.name),f
    assert sha(native/'video_input_lut.sv')==sha(board/'video_input_lut.sv')
    return dict(changed_native_cells=count,retained_native_cells=len(a),pin_changes=dict(categories),protected_input_byte_identical=True)


def warnings(board):
    parent=ROOT/'tmp/frame-execution/connected-board-video-service-probe58-color-exposure'
    def read(d):
        rows=[]
        for l in (d/'outflow/ti60f225_oob.warn.log').read_text().splitlines():
            if l.startswith('['):rows.append(re.sub(r'\([^()]*:\d+\)','(SOURCE)',l))
        return collections.Counter(rows)
    x,y=read(parent),read(board);added,removed=y-x,x-y
    # EFX-0200 displays ten optimizer examples, not all optimized nets. Only
    # unchanged protected blocks may differ in this truncated example list.
    for line in list(added)+list(removed):
        assert line.startswith('[EFX-0200 WARNING] Removing redundant signal : '),line
        name=line.split(' : ',1)[1].replace('protected_inst/','')
        assert name.startswith(('u_ddr3_top/','dvi_encoder_m0/','inst_efx_csi2_rx/')),line
    assert sum(added.values())==sum(removed.values())==10
    return dict(current_count=sum(y.values()),identical_count=sum((x&y).values()),
                changed_protected_optimizer_examples=dict(added),removed_examples=dict(removed),zero_warning_claim=False)


def compare_released_map(board):
    """Compare retained cells against the already validated map, not pre-map
    high-level constant outputs (the sparse-weight decoder predates this edit)."""
    from probe_native_open_ports import compare_boundary
    from audit_native_roundtrip import tokens
    from audit_local_reset_implementation import norm
    meta=json.loads((board/'inputs.json').read_text());parent=Path(meta['parent']);native=Path(meta['native'])
    key,src,new,sp,np=pickle.loads((board/'retained-audit-parse.pkl').read_bytes())
    h,old,op=pickle.loads((parent/'color-map-parse.pkl').read_bytes());assert h==sha(parent/'outflow/ti60f225_oob.map.v')
    proof=json.loads((native/'input-isp-edit.json').read_text())
    # All original public native instances are selected. Existing generated
    # sparse-weight constants come from the released67 map's validated proof.
    assert set(src)-set(old)==set(src)-set(new)
    mapped_folds(src,new,sp,np,(board/'native_source.v').read_text())
    a={n:(old[n][0],{p:list(v) for p,v in old[n][1].items()}) for n in src}
    pa={n:op[n] for n in a}
    for n,ps in proof['edits'].items():
        for p,changes in ps.items():
            for change in changes:
                x,y=tuple(change['old']),tuple(change['new'])
                assert x in a[n][1][p],(n,p,x)
                a[n][1][p]=[y if t==x else t for t in a[n][1][p]]
    forward={};reverse={};conflicts=[]
    for n,(kind,pins) in a.items():
        assert new[n][0]==kind and pins.keys()==new[n][1].keys(),n
        if kind!='EFX_LUT4':assert pa[n]==np[n],('params',n)
        for p,xs in pins.items():
            ys=new[n][1][p];assert len(xs)==len(ys),(n,p,xs,ys)
            for x,y in zip(xs,ys):
                if x[0]=='literal' or y[0]=='literal':
                    assert x==y,('literal',n,p,x,y);continue
                if x in forward and forward[x]!=y:conflicts.append((n,p,x,y,forward[x]))
                if y in reverse and reverse[y]!=x:conflicts.append((n,p,x,y,reverse[y]))
                forward[x]=y;reverse[y]=x
    (board/'input-isp-map-inspection.json').write_text(json.dumps(dict(conflicts=conflicts[:100],count=len(conflicts)),indent=2))
    assert not conflicts,('net_bijection',len(conflicts),conflicts[:5])
    for n,(kind,pins) in a.items():
        translate=lambda ps,alias:{p:('' if not vs else render_token(alias.get(vs[0],vs[0])) if len(vs)==1 else '{'+','.join(render_token(alias.get(t,t)) for t in vs)+'}') for p,vs in ps.items()}
        x=translate(pins,forward);y=translate(new[n][1],{})
        if kind=='EFX_LUT4':assert lut_function(x,pa[n]['LUTMASK'][1])==lut_function(y,np[n]['LUTMASK'][1]),n
        else:assert x==y,n
    renamed={tokens(norm(render_token(x)))[0]:tokens(norm(render_token(y)))[0] for x,y in forward.items()}
    boundary=compare_boundary((parent/'outflow/ti60f225_oob.map.v').read_text(),(board/'outflow/ti60f225_oob.map.v').read_text(),renamed)
    report=dict(status='INPUT_ISP_RELEASED_MAP_RETAINED_PASS',retained=len(a),source_sha256=key[0],map_sha256=key[1],parent_map_sha256=h,protected_boundary=boundary,
        scope='All retained public native cells against released67 map after 179 explicit cuts; unchanged support RTL; official encrypted internals not independently proved')
    (board/'input-isp-retained-map.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);args=p.parse_args();board=args.board.resolve()
    start=time.monotonic();result=json.loads((board/'result.json').read_text());assert result['status']=='VIDEO_SERVICE_COMPLETE_NATIVE_SYNTHESIS_PROBE_PASS'
    source_audit=source_delta(board)
    mapped=compare_released_map(board)
    warning_audit=warnings(board)
    native=Path(json.loads((board/'inputs.json').read_text())['native'])
    parent=Path(json.loads((board/'inputs.json').read_text())['parent'])
    evidence=[Path(__file__),board/'native_source.v',board/'inputs.json',board/'result.json',board/'input-isp-retained-map.json',native/'input-isp-edit.json',board/'outflow/ti60f225_oob.warn.log',parent/'color-map-audit.json',parent/'outflow/ti60f225_oob.map.v']
    out=dict(status='INPUT_ISP_SOURCE_AND_RETAINED_MAP_PASS',source=source_audit,retained=mapped['retained'],
       equivalent_folded_cells=['LUT__133287','LUT__140273'],unloaded_removed_cells=['cnn_run_seen~FF'],warnings=warning_audit,
       source_sha256=sha(board/'native_source.v'),map_sha256=result['map_sha256'],vdb_sha256=result['vdb_sha256'],
       evidence={str(f):sha(f) for f in evidence},hardware_access=False,board_release=False,seconds=time.monotonic()-start)
    (board/'input-isp-map-audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('evidence','warnings')}))
if __name__=='__main__':main()
