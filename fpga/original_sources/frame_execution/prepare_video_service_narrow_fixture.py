"""Adapt the proved native-reference transport fixture to all32 legacy words."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 src=ROOT/'tmp/frame-execution/video-service-compact-ram-fixture2';out=a.root.resolve();native=a.native.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 meta=json.loads((src/'inputs.json').read_text())
 for f,h in meta['files'].items():assert sha(f)==h,f
 out.mkdir()
 for f in src.iterdir():
  if f.suffix in ('.v','.sv'):shutil.copy2(f,out/f.name)
 for name in ('old.sv','new.sv'):
  f=out/name;s=f.read_text()
  s=s.replace('input wire[1:0]service_page_part','input wire[3:0]service_page_part').replace('input wire[182:0]service_observation_packet','input wire[86:0]service_observation_packet')
  if name=='new.sv':
   changes=[('[2:0] service_legacy_part','[4:0] service_legacy_part'),('[191:0] service_compact_packet','[95:0] service_compact_packet'),
    ("16'd0,8'd4,service_page_sequence,29'h989e000","16'd0,8'd5,service_page_sequence,27'h2627800"),
    ('service_legacy_part*128+:128','service_legacy_part*32+:32'),
    ("{9'd0,service_observation_packet[182:176],8'd4,service_observation_packet[167:0]}","{9'd0,service_observation_packet[86:80],8'd5,service_observation_packet[71:0]}")]
   for old,new in changes:assert s.count(old)==1,old;s=s.replace(old,new)
  f.write_text(s)
 s=(out/'tb.sv').read_text().replace('[1:0]service_page_part','[3:0]service_page_part').replace('[182:0]service_observation_packet','[86:0]service_observation_packet')
 s=s.replace('[191:0]expected_','[95:0]expected_').replace('[2:0]source_index','[4:0]source_index')
 s=s.replace("8'd4,8'd0","8'd5,8'd0").replace(",128'd0};",",32'd0};")
 s=s.replace("{9'd0,service_observation_packet[182:176],8'd4,service_observation_packet[167:0]}","{9'd0,service_observation_packet[86:80],8'd5,service_observation_packet[71:0]}")
 start=s.index('    if(record[1023:256]');end=s.index('\n   end',start)
 s=s[:start]+'''    if(record[1023:160]!==0)$fatal(1,"narrow tail nonzero");
    if(record[159:96]!==expected_scan[95:32])$fatal(1,"narrow meta mismatch %h %h",record[159:96],expected_scan[95:32]);
    if(scan_legacy)begin
     if(scan_index>=30)begin
      if(record[95:64]!==0)$fatal(1,"legacy zero padding");
     end else if(record[95:64]!==old_record[(64+scan_index*32)+:32])$fatal(1,"legacy word mismatch %d",scan_index);
    end else if(record[95:64]!==expected_scan[31:0])$fatal(1,"service payload changed during scan");'''+s[end:]
 s=s.replace('j<24','j<96')
 s=s.replace("8'd2,8'd17","8'd5,8'd17").replace("32'hdeadbeef,32'h12345678,32'h87654321,32'hcafefeed","32'hcafefeed")
 s=s.replace("8'd20","8'd92").replace('26 completions','98 completions').replace('completed=26','completed=98')
 (out/'tb.sv').write_text(s)
 for name in ('video_dualclock_sample.sv','video_snapshot_serializer.sv'):shutil.copy2(native/name,out/name)
 files={str(f):sha(f) for f in out.iterdir()};files.update({str(Path(__file__)):sha(__file__),str(src/'inputs.json'):sha(src/'inputs.json')})
 (out/'inputs.json').write_text(json.dumps(dict(files=files,parent=str(src),native_graft_sha256=sha(native/'graft.json'),geometry='96-bit source/160-bit scan image/1024-bit external scan; all32 legacy words',hardware_access=False),indent=2))
 print('NARROW_NATIVE_TRANSPORT_FIXTURE_PREPARED')
if __name__=='__main__':main()
