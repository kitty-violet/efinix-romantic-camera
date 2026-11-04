"""Locate actual AXI mismatch patterns without relying on the vendor checker."""
import argparse,csv,json
from pathlib import Path
from run_units import sha

def chunks(word):return [word[n:n+4] for n in range(28,-1,-4)]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();path=root/'axi_transfers.csv'
    with path.open() as f:rows=list(csv.DictReader(f))
    writes={int(row['address_or_index'],16):row['data'].lower() for row in rows if row['channel']=='W'}
    reads=[row for row in rows if row['channel']=='R']
    mismatches=[];sequence={index*8+k:value for index,word in writes.items() for k,value in enumerate(chunks(word))}
    scores=[]
    for offset in range(-32,33):
        hits=compared=unknown=0
        for row in reads:
            index=int(row['address_or_index'],16)
            for k,actual in enumerate(chunks(row['data'].lower())):
                expected=sequence.get(index*8+k+offset)
                if expected is None:continue
                compared+=1
                if any(c in actual for c in 'xz'):unknown+=1
                elif actual==expected:hits+=1
        scores.append(dict(offset_16bit_words=offset,matches=hits,compared=compared,unknown=unknown,ratio=hits/compared if compared else 0))
    for row in reads:
        index=int(row['address_or_index'],16);actual=row['data'].lower();expected=writes[index]
        if row['expected'].lower()!=expected:raise ValueError('Diagnostic expected field differs from accepted W')
        if actual!=expected:mismatches.append(dict(index=index,actual=actual,expected=expected))
    scores.sort(key=lambda x:(-x['ratio'],abs(x['offset_16bit_words'])))
    report=dict(status='AXI_WORD_DIFFERENCE_ANALYZED' if len(writes)==256 and len(reads)==256 else 'PARTIAL_AXI_WORD_TRACE',
        writes=len(writes),reads=len(reads),mismatch_words=len(mismatches),first_mismatches=mismatches[:8],
        best_stream_offsets=scores[:5],csv_sha256=sha(path),board_release=False,
        note='Offset correlation is evidence to guide diagnosis, not proof of controller implementation.')
    (root/'word_analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
