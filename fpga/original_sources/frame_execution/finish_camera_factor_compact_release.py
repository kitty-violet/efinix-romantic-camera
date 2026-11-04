"""Same release gates with exact compact decoder depth and bound dependencies."""
from pathlib import Path
COMPACT_RELEASE_BASE=Path(__file__).with_name('finish_camera_factor_balanced_release.py')
code=COMPACT_RELEASE_BASE.read_text().replace('camera_factor_balanced','camera_factor_compact').replace('camera-factor-balanced','camera-factor-compact')
needle="exec(compile(code,str(BASE_SCRIPT)+':factor-all-gates','exec'),globals())"
assert code.count(needle)==1
inject="code=code.replace(\"t['new_divider_depth']==8\",\"t['new_divider_depth']==10\").replace('exact12-to8-level','exact12-to10-level')\n"
old=" current=bind(SHORTS/'result.json')"
new=" verify(bind(ROOT/'tmp/board-test/camera-factor-compact-generators1.json')['evidence'])\n"+old
inject+=f"code=code.replace({old!r},{new!r})\n"
code=code.replace(needle,inject+needle)
exec(compile(code,str(COMPACT_RELEASE_BASE)+':compact-all-gates','exec'),globals())
