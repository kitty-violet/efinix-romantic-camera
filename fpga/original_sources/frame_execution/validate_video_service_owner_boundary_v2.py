"""Resolve owner cuts from the proved CURRENT native ports, not historic pins."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_owner_boundary.py');code=source.read_text()
start=code.index("  old={n:(k,{p:list(v)")
end=code.index('  od,ol=connections(old)',start)
code=code[:start]+'''  retained=json.loads((board/'retained-map-audit.json').read_text())
  assert retained['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS'
  binding_key,current_native,_,_,_=pickle.loads((board/'retained-audit-parse.pkl').read_bytes())
  assert binding_key[0]==sha(board/'native_source.v') and binding_key[1]==sha(src)
  assert retained['source_sha256']==binding_key[0] and retained['map_sha256']==binding_key[1]
  old={n:(k,{pin:list(ts) for pin,ts in pins.items()}) for n,(k,pins) in current_native.items()}
  (out/'current-binding.json').write_text(json.dumps(dict(status='CURRENT_PROVED_NATIVE_PORT_BINDING',source_sha256=binding_key[0],map_sha256=binding_key[1],audit_sha256=sha(board/'retained-map-audit.json'),checker_sha256=sha(__file__),parent_checker_sha256=sha(HERE/'validate_video_service_owner_boundary.py')),indent=2))
''' +code[end:]
needle="  (out/'inputs.json').write_text("
assert code.count(needle)==1
code=code.replace(needle,"  files.update({str(HERE/'validate_video_service_owner_boundary.py'):sha(HERE/'validate_video_service_owner_boundary.py'),str(out/'current-binding.json'):sha(out/'current-binding.json')})\n"+needle)
exec(compile(code,str(source)+':current-native-pin-binding','exec'),globals())
