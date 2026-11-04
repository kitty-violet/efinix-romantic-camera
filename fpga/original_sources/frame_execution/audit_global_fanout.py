"""Compare actual public primitive input reference counts; exclude real clocks."""
import argparse,collections,json,re
from pathlib import Path
from run_units import sha

OUTPUTS={'EFX_FF':{'Q'},'EFX_LUT4':{'O'},'EFX_COMB4':{'O','P'},'EFX_ADD':{'O','CO'},
 'EFX_SRL8':{'Q','Q7'},'EFX_DSP24':{'O','CASCOUT'},'EFX_DSP48':{'O','CASCOUT'},
 'EFX_RAM10':{'RDATA'},'EFX_DPRAM10':{'RDATA_A','RDATA_B'}}

def inspect(path):
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',path.read_text(),flags=re.S)
    loads=collections.defaultdict(set);unknown=collections.Counter();drivers=collections.defaultdict(set)
    for kind,name,body in re.findall(r'^\s*(EFX_\w+)\s+(\S+)\s*\((.*?)\)\s*;',text,re.M|re.S):
        if kind not in OUTPUTS:unknown[kind]+=1;continue
        for port,expr in re.findall(r'\.(\w+)\(([^()]*)\)',body):
            expr=re.sub(r"\d+'[sS]?[bBoOhHdD][0-9a-fA-F_xXzZ?]+",'',expr)
            refs={re.sub(r'\s+','',r).lstrip('\\') for r in re.findall(
                r'\\[^\s{}(),]+(?:\s+\[\d+(?::\d+)?\])?|[A-Za-z_]\w*(?:\[\d+(?::\d+)?\])?',expr)}
            for signal in refs:
                if port in OUTPUTS[kind]:drivers[signal].add(name.lstrip('\\'))
                else:loads[signal].add((name.lstrip('\\'),port))
    rows=[]
    for net,pins in loads.items():
        if all(p in ('CLK','CLKA','CLKB','RCLK','WCLK') for _,p in pins):continue
        if len(pins)>100:rows.append(dict(net=net,count=len(pins),ports=dict(collections.Counter(p for _,p in pins)),drivers=sorted(drivers[net])))
    rows.sort(key=lambda r:r['count'],reverse=True)
    return dict(map=str(path),sha256=sha(path),over100=rows,over100_count=len(rows),
                excess_references=sum(r['count']-100 for r in rows),unknown=dict(unknown))

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=dict(status='ACTUAL_MAP_FANOUT_COMPARISON_NOT_TIMING_PASS',base=inspect(a.base),candidate=inspect(a.candidate),
      limits='Counts are public primitive input-port references; vectors/aliases not expanded, reset/control mixed; no post-route fanout claim.')
    assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:dict(over100=result[k]['over100_count'],excess=result[k]['excess_references'],top=result[k]['over100'][:5]) for k in ('base','candidate')}))
if __name__=='__main__':main()
