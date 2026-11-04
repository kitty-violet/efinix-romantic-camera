"""Full current page oracle, binding the one intentionally replaced producer."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_narrow_mapped_pages.py')
code=source.read_text()
anchor="exec(compile(code,str(source)+':narrow-word-oracle','exec'),globals())"
patch='''
code=code.replace('from video_service_serializer_native import P','from video_service_serializer_native import P\\nfrom same_edge_observation_binding import bind_apb')
old=" roots=capture_roots(g,ps)";assert code.count(old)==1
code=code.replace(old,old+"\\n apb_binding=bind_apb(board,a.native,g,ps)")
old=' def mapped(t):';assert code.count(old)==1
code=code.replace(old,old+"\\n  if t==net('apb_enable'):return tuple(apb_binding['token'])")
old=' result=dict(status=';assert code.count(old)==1
code=code.replace(old,' result=dict(apb_observation_binding=apb_binding,status=')
'''
assert code.count(anchor)==1
code=code.replace(anchor,patch+'\n'+anchor)
exec(compile(code,str(source)+':proved-same-edge-producer','exec'),globals())
