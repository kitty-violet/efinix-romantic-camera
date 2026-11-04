"""Verify the complete delivered source and rebuild all runtime data in memory."""
from pathlib import Path
import ast,hashlib,importlib.util,json,re,struct,sys,xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'network'))
from coefficients import convert
from verify_package import load,oracle
import style_catalog as catalog
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=json.loads((ROOT/'MANIFEST.json').read_text('utf-8'))
 for name,h in manifest['files'].items():assert sha(ROOT/name)==h,name
 project=ROOT/'project';tree=ET.parse(project/'ti60f225_oob.xml');files=[]
 for e in tree.iter():
  if e.tag.split('}')[-1] in ('design_file','sdc_file','peri_file','default_peri_file'):
   p=project/e.attrib['name'];assert p.is_file() and p.resolve().is_relative_to(project.resolve());files.append(p)
 for p in files:
  if p.suffix not in ('.v','.sv'):continue
  # Match ASCII HDL directives without rewriting original comment encodings.
  text=p.read_bytes().decode('latin1')
  for name in re.findall(r'`include\s+"([^"]+)"|\$readmem[hb]\s*\(\s*"([^"]+)"',text):
   assert (project/next(x for x in name if x)).is_file(),(p,name)
 fw=(ROOT/'firmware/prebuilt/control.bin').read_bytes();fw+=bytes(8192-len(fw))
 for i in range(4):
  p=project/f'EfxSapphireSoc.v_toplevel_system_ramA_logic_ram_symbol{i}.bin'
  assert bytes(int(x,2) for x in p.read_text().split())==fw[i::4]
 rtl=ROOT/'network/rtl';core=json.loads((rtl/'result.json').read_text())
 adjustments=json.loads((rtl/'portable-adjustments.json').read_text())
 for name,h in core['inputs'].items():
  expected=h
  if name in adjustments:
   a=adjustments[name];assert a['before']==expected==sha(ROOT/a['original'])
   old=(ROOT/a['original']).read_text()
   rebuilt=old.replace(' reg fused;',' wire start;\n reg fused;').replace(' wire start=cmd_valid && cmd_ready;',' assign start=cmd_valid && cmd_ready;')
   assert rebuilt==(rtl/name).read_text();expected=a['after']
  assert sha(rtl/name)==expected,name
 rt=ET.parse(rtl/'cnn_production_frame.xml')
 refs=[e.attrib['name'] for e in rt.iter() if e.tag.split('}')[-1]=='design_file']
 for n in refs:assert (rtl/n).is_file(),n
 results=[]
 for i in range(5):
  folder=ROOT/'network/models'/f'slot{i}';graph,params=load(folder)
  assert graph['schema']=='S4R1_8_32_8_V1' and len(graph['nodes'])==13 and len(params)==11
  known={'input'}
  for n in graph['nodes']:assert all(k in known for k in n['inputs']);known.add(n['id'])
  resident,metadata=convert(folder);assert resident==(folder/'resident-padded.bin').read_bytes()
  count=0
  for v in json.loads((folder/'reference-vectors.json').read_text())['vectors']:
   assert oracle(graph,params,v['input_u8'])==v['output_i8'];count+=len(v['output_i8'])*len(v['output_i8'][0])*3
  results.append({'slot':i,'conversion_checks':metadata['independent_weight_bias_multiplier_checks'],'reference_scalars':count})
 prog=ROOT/'network/programs';schema=json.loads((prog/'schema.json').read_text());rows=json.loads((prog/'descriptor-fields.json').read_text());counts=[]
 for j,name in enumerate(('compute','dma')):
  size,bits,version=(64,8,0x55) if j==0 else (16,4,3);out=bytearray()
  for values in rows[name]:
   word=offset=0
   for field,width in schema[j]:
    v=values[field];assert 0<=v<(1<<width);word|=v<<offset;offset+=width
   assert offset<=size*8-bits;word|=version<<(size*8-bits);out.extend(word.to_bytes(size,'little'))
  assert out==(prog/f'{name}.bin').read_bytes();counts.append(len(rows[name]))
 assert counts==[6546,8277]
 raw=(ROOT/'network/runtime_data/flash-user-data.bin').read_bytes();header=catalog.validate(raw)
 isp=raw[header[10]-catalog.CATALOG_BASE:header[10]-catalog.CATALOG_BASE+768]
 # Preserve exact slot names/tags from the original directory manifest.
 meta=json.loads((ROOT/'network/runtime_data/manifest.json').read_text())
 entries=meta['styles']
 names=[r['name'] for r in entries]
 styles=[dict(name=name,tag=i+1,compute=(prog/'compute.bin').read_bytes(),coefficients=(ROOT/'network/models'/f'slot{i}/resident-padded.bin').read_bytes()) for i,name in enumerate(names)]
 rebuilt,_=catalog.build(styles,(prog/'dma.bin').read_bytes(),isp)
 assert rebuilt==raw,'Runtime directory/data mismatch'
 for p in [ROOT/'network/model.py',*ROOT.joinpath('tools').glob('*.py')]:ast.parse(p.read_text('utf-8'),filename=str(p))
 print(json.dumps({'status':'COMPLETE_SOURCE_AND_NETWORK_PACKAGE_PASS','files':len(manifest['files']),'project_references':len(files),'core_design_files':len(refs),'styles':results,'descriptor_counts':counts,'runtime_bytes':len(raw),'rom_bytes':len(fw)},indent=2))
if __name__=='__main__':main()
