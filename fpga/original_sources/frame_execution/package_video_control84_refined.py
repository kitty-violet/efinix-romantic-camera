"""84 package gates plus exact provenance of the reviewed detailed refinement."""
from pathlib import Path

source=Path(__file__).with_name('package_video_control84.py')
code=source.read_text(encoding='utf8')
needle="bind(Path(__file__));bind(Path(__file__).with_name('video_control84_physical.py'))"
assert code.count(needle)==1
code=code.replace(needle,needle+"\n bind(source)\n embedded(bind(physical/'refinement-inputs.json'))\n embedded(bind(physical/'refinement-result.json'))\n margin=json.loads((physical/'physical_inputs.json').read_text())['setup_margin']\n embedded(bind(margin['proof'],margin['proof_sha256']))")
needle='本包自身原约束WNS';assert code.count(needle)==1
code=code.replace(needle,'本包保留84冻结映射，以C2/seed1在独立副本进行详细布局及布线优化，优化期间提前预留0.300ns建立时间。最终STA已恢复原始完整约束，额外余量不是本次验收门槛。\n\n本包自身原约束WNS')
needle="implementation_basis='fresh84 map/place/route with scoped capture/bridge change and control factoring and accepted82 sensorROM; separate unchanged curve80 upload'";assert code.count(needle)==1
code=code.replace(needle,"implementation_basis='84 verified source/map; C2seed1 setup-aware detailed-placement refinement and fresh route; original-constraint acceptance; capture32/independentDDRbanks/controlfactor and82 sensorROM; unchanged curve80 upload'")
exec(compile(code,str(source)+':refined-provenance','exec'),globals())
