"""Four-state finite program transport with independent binary/CRC oracle."""
import argparse,json,re,shutil,zlib
from pathlib import Path
from run_units import HERE,ROOT,MS,EF,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--descriptors',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--limit',type=int);p.add_argument('--deep-backpressure',action='store_true');a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
records=[]
if a.source and 'EFX_SRL8' in a.source.read_text():
 src=EF/'sim_models/verilog/efx_srl8.v';dst=root/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
for name in ('frame_program_fetch.sv','tb_frame_program_fetch.sv'):
 src=a.source.resolve() if a.source and name=='frame_program_fetch.sv' else HERE/name
 assert src.is_relative_to(ROOT)
 dst=root/name
 if name=='tb_frame_program_fetch.sv' and (a.limit or a.deep_backpressure):
  body=src.read_text()
  if a.limit:
   assert 8<=a.limit<=8789
   body=body.replace('8788',str(a.limit-1)).replace('10439',str(a.limit-1)).replace('8789',str(a.limit)).replace('10440',str(a.limit)).replace('19229',str(a.limit*2)).replace('45596',str(a.limit*5))
  if a.deep_backpressure:
   body=body.replace('cready<=rng[7:4]!=0;dready<=rng[11:8]!=0;', 'cready<=(ticks%113>67 && rng[7:4]!=0) || (!early_pop_sent && dut.ccount==1 && (|dut.partial_compute));dready<=ticks%97>45 && rng[11:8]!=0;')
   body=body.replace(' reg arheld=0,chold=0,dhold=0;', ''' reg early_pop_sent=0;
 integer full_c=0,full_d=0,partial_hold=0,partial_pop=0;
 always @(posedge clk)begin
  if(reset)begin full_c<=0;full_d<=0;partial_hold<=0;partial_pop<=0;early_pop_sent<=0;end
  else begin
   if(dut.ccount==2)full_c<=full_c+1;
   if(dut.dcount==2)full_d<=full_d+1;
   if(cvalid && !cready && (|dut.partial_compute))partial_hold<=partial_hold+1;
   if(cvalid && cready && (|dut.partial_compute))begin partial_pop<=partial_pop+1;early_pop_sent<=1;end
  end
 end
 reg arheld=0,chold=0,dhold=0;''')
   body=body.replace('   $display("PASS PROGRAM_NORMAL', '''   if(full_c==0 || full_d==0 || partial_hold==0 || partial_pop==0)$fatal(1,"Missing full/partial backpressure coverage");
   $display("COVER PROGRAM_SRL full_c=%0d full_d=%0d partial_hold=%0d partial_pop=%0d",full_c,full_d,partial_hold,partial_pop);
   $display("PASS PROGRAM_NORMAL''')
  dst.write_text(body)
 else:shutil.copy2(src,dst)
 records.append(dict(source=str(src),source_sha256=sha(src),snapshot=str(dst),sha256=sha(dst)))
manifest=json.loads((a.descriptors/'result.json').read_text());crc={}
for stream,width in (('compute',64),('dma',16)):
 f=a.descriptors/(stream+'.bin');expected=manifest['streams'][stream]['sha256'] if 'streams' in manifest else manifest['files'][stream+'.bin'];assert sha(f)==expected
 data=f.read_bytes()
 if a.limit:data=data[:a.limit*width]
 crc[stream]=zlib.crc32(data)
 (root/(stream+'.hex')).write_text(''.join(format(int.from_bytes(data[i:i+width],'little'),'0'+str(width*2)+'x')+'\n' for i in range(0,len(data),width)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,descriptor_manifest_sha256=sha(a.descriptors/'result.json'),crc32=crc),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
assert 'Errors: 0, Warnings: 0' in body
body,secs=run([MS/'vsim.exe','-c','work.tb_frame_program_fetch','-gCCRC='+str(crc['compute']),'-gDCRC='+str(crc['dma']),'+VECTORS='+root.as_posix(),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=300)
lines=re.findall(r'^# (PASS PROGRAM_.*)$',body,re.M)
assert len(lines)==6 and 'PASS PROGRAM_FETCH' in body and 'Errors: 0, Warnings: 0' in body
for r in records:assert sha(r['source'])==r.get('source_sha256',r['sha256']) and sha(r['snapshot'])==r['sha256']
result=dict(status='PROGRAM_FETCH_FOURSTATE_PASS',lines=lines,seconds=secs,records_per_stream=a.limit,deep_backpressure=a.deep_backpressure,coverage=re.findall(r'^# (COVER PROGRAM_SRL.*)$',body,re.M),inputs_sha256=sha(root/'inputs.json'),simulation_sha256=sha(root/'simulation.log'),full_system=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

