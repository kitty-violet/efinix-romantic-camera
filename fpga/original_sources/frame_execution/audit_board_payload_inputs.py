"""Bind simulation readmemh inputs to the validated bounded boot payload."""
import argparse,json
from pathlib import Path
from run_units import sha

def audit(root,payload):
 meta=json.loads((payload/'payload.json').read_text());rows=[]
 for item in meta['payloads']:
  source=payload/item['file'];assert sha(source)==item['sha256']
  target=root/(source.stem.replace('-','_')+'.hex')
  decoded=b''.join(int(word,16).to_bytes(16,'little') for word in target.read_text().split())
  assert decoded==source.read_bytes() and len(decoded)==item['bytes'],target
  rows.append(dict(snapshot=str(target),sha256=sha(target),binary_source=str(source),binary_sha256=item['sha256'],bytes=len(decoded)))
 return dict(status='BOARD_PAYLOAD_INPUTS_MATCH',payload_manifest_sha256=sha(payload/'payload.json'),
  payloads=rows,total_bytes=sum(r['bytes'] for r in rows),network_contract_sha256=meta['network_contract_sha256'],diagnostic_weights=meta['diagnostic_weights'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--payload',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();out=root/'payload_integrity_audit.json';assert not out.exists()
 result=audit(root,a.payload.resolve());out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
