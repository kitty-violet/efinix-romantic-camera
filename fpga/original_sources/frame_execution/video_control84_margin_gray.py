"""Run all existing CDC/Gray checks using the bound original84 constraints."""
from pathlib import Path

source = Path(__file__).with_name('video_service_gray.py')
code = source.read_text(encoding='utf8')
for old, new in [
    ('audit_video_service_compact_routed_cdc', 'audit_video_service_narrow_split_cdc'),
    ('audit_video_service_compact_jtag', 'audit_video_service_narrow_jtag'),
]:
    assert old in code
    code = code.replace(old, new)
old = '"canonical=b/\'ti60f225_oob.sdc\'\\nassert meta[\'setup_margin\'] is None\\nassert sha(canonical)==next(x[\'sha256\'] for x in meta[\'inputs\'] if Path(x[\'snapshot\'])==canonical)"'
new_body = """margin=meta['setup_margin']
assert margin['extra_setup_ns']==.3 and margin['extra_hold_ns']==0
canonical=Path(margin['canonical_sdc'])
assert sha(canonical)==margin['canonical_sdc_sha256']=='464ef739d92e3fb1d2b15018863be99636efc1cae42ca581a0526f305c4be262'
assert sha(b/'ti60f225_oob.sdc')==margin['optimization_sdc_sha256']
assert sha(margin['proof'])==margin['proof_sha256']"""
assert code.count(old) == 1
code = code.replace(old, repr(new_body))
exec(compile(code, str(source)+':bound-original84-margin', 'exec'), globals())
