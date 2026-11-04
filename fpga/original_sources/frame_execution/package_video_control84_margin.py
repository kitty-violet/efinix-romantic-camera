"""Existing84 release gates with explicit same-placement reroute provenance."""
from pathlib import Path

source = Path(__file__).with_name('package_video_control84.py')
code = source.read_text(encoding='utf8')
needle = "bind(Path(__file__));bind(Path(__file__).with_name('video_control84_physical.py'))"
assert code.count(needle) == 1
code = code.replace(needle, needle + "\n bind(source)\n embedded(bind(physical/'geometry-reuse-proof.json'))\n margin=json.loads((physical/'physical_inputs.json').read_text())['setup_margin']\n embedded(bind(margin['proof'],margin['proof_sha256']))")
needle = '本包自身原约束WNS'
assert code.count(needle) == 1
code = code.replace(needle, '本包沿用84冻结映射和布局，仅在独立副本以额外0.300ns建立时间优化压力重布线。最终STA已恢复原始完整约束，额外余量不是本次验收门槛。\n\n本包自身原约束WNS')
needle = "implementation_basis='fresh84 map/place/route with scoped capture/bridge change and control factoring and accepted82 sensorROM; separate unchanged curve80 upload'"
assert code.count(needle) == 1
code = code.replace(needle, "implementation_basis='84 verified source/map and frozen placement; independent setup-pressure reroute with original-constraint acceptance; capture32/independentDDRbanks/controlfactor and82 sensorROM retained; unchanged curve80 upload'")
exec(compile(code, str(source)+':margin-provenance', 'exec'), globals())
