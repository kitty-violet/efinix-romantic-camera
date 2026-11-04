"""Keep all short gates; reuse frozen hashes in each child as in the parent."""
from pathlib import Path
source=Path(__file__).with_name('continuous_srl_register_flow.py')
body=source.read_text(encoding='utf8')
needle="exec(compile(code,str(source)+':srl-register-shorts','exec'),globals())"
assert body.count(needle)==1
injection="""code=code.replace('run-with-locked-evidence.py','run-with-continuous-locked-evidence.py')
code=code.replace("root/'map-audit'],360)","root/'map-audit'],600)")
"""
body=body.replace(needle,injection+needle)
exec(compile(body,str(source)+':locked-children','exec'),globals())
