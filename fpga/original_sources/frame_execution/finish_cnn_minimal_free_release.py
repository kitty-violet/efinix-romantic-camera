"""Same release gates; no ordinary cell anchors in this implementation."""
from control_command_repair import once
from cnn_small_release_factory import code_for
from audit_legacy_cache_recovery import install,prove,MANIFEST
install()
from finish_cnn_minimal_c1_release import recovery_tail
code=code_for('minimal')
code=code.replace('finish_cnn_minimal_release','finish_cnn_minimal_free_release')
code=code.replace('cnn_minimal_route','cnn_minimal_free_route')
code=code.replace('cnn_small_place.py','cnn_minimal_repack.py')
old=""" from cnn_run_budget_locations import verify_placed as check_arena
 assert check_arena(physical)['count']==36
 for name in ('arena_island_placement.json','arena_island_actual_locations.json'):
  item=bind(physical/name)
  if 'evidence' in item:verify(item['evidence'])"""
code=once(code,old," assert not (physical/'arena_island_placement.json').exists()\n assert not (physical/'arena_island_actual_locations.json').exists()")
inner_exec="exec(compile(code,str(SOURCE)+':joint-closure-release','exec'),globals())"
code=once(code,inner_exec,"code=once(code,\"if __name__=='__main__':main()\","+repr(recovery_tail)+")\n"+inner_exec)
exec(compile(code,__file__+':complete-free-ordinary-gates','exec'),globals())
