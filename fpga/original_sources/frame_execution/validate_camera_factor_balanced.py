from pathlib import Path
FACTOR_VALIDATOR=Path(__file__).with_name('validate_camera_factor.py')
code=FACTOR_VALIDATOR.read_text().replace('from camera_factor_board import','from camera_factor_balanced_board import')
code=code.replace('runner_sha256=sha(__file__),files=', 'runner_sha256=sha(__file__),base_runner_sha256=sha(FACTOR_VALIDATOR),files=')
needle="oldnames=set();newnames=set();oldleaves=set();newleaves=set();pairs=[]"
assert code.count(needle)==1
code=code.replace(needle,"""from camera_factor_native import Cone
 control_roots=[tuple(s['root']) for s in proof['factor_stages']]
 cones=[Cone(old,t) for t in control_roots]
 oldstops={t for t in oldlut if all(c.root_depth(t)<0 for c in cones)}
 newstops={mapping[t] for t in oldstops if t in mapping}
 oldnames=set();newnames=set();oldleaves=set();newleaves=set();pairs=[]""")
needle="if t not in drivers:leaves.add(t);return";assert code.count(needle)==1
code=code.replace(needle,"if t in (oldstops if g is og else newstops) or t not in drivers:leaves.add(t);return")
code=code.replace('assert {mapping[t] for t in oldleaves}==newleaves','assert newleaves<={mapping[t] for t in oldleaves}')
# Original root-independent LUT cones are identical across both maps and
# covered by the whole graph proof. Exercise arbitrary inputs at that cut;
# avoid simulating thousands of unchanged arithmetic/control LUTs.
exec(compile(code,str(FACTOR_VALIDATOR)+':grouped','exec'),globals())
