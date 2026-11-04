"""Exact two-bit decomposition plus the existing group-preserving factors."""
import json,sys
from pathlib import Path
from run_units import HERE,sha
BALANCED_GENERATOR=HERE/'camera_factor_balanced_native.py'
namespace={'__name__':'compact_factor_library','__file__':__file__}
exec(compile(BALANCED_GENERATOR.read_text(),str(BALANCED_GENERATOR)+':compact','exec'),namespace)

def divider_only(text,placement):
 source=HERE/'camera_joint_native.py';code=source.read_text()
 replacements={
  'if len(vars)<=7:return [table(vars,t) for t in tabs]':'if len(vars)<=5:return [table(vars,t) for t in tabs]',
  'width=4;bits=1<<width':'width=2;bits=1<<width',
  'for n,tok,groups,fixed in sorted(candidates,key=lambda r:(-len(loads[r[1]]),r[0])):':'for n,tok,groups,fixed in []:',
  'assert 0<len(clone_orig)<=1200,len(clone_orig)':'assert len(clone_orig)==0',
 }
 for before,after in replacements.items():
  assert code.count(before)==1,before;code=code.replace(before,after)
 ns={'__name__':'compact_divider_library','__file__':str(source)}
 exec(compile(code,str(source)+':two-bit-exact-decomposition','exec'),ns)
 text,proof=ns['transform'](text,placement)
 assert proof['divider_added_cells']==68 and proof['old_divider_depth']==12 and proof['new_divider_depth']==10
 return text,proof

namespace['divider_only']=divider_only
def main():
 namespace['main']()
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();p=root/'result.json';r=json.loads(p.read_text())
 r['inputs'][str(BALANCED_GENERATOR)]=sha(BALANCED_GENERATOR)
 r['decomposition']=dict(width=2,leaf_limit=5,exact_patterns=8192,late_input_lut_depths=[2,2,4],maximum_internal_fanout=9)
 p.write_text(json.dumps(r,indent=2),encoding='utf8')
if __name__=='__main__':main()
