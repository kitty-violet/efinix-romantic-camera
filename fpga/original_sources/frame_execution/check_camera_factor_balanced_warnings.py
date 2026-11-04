"""Prove each current dangling endpoint, retain all inherited checks."""
from pathlib import Path
source=Path(__file__).with_name('check_camera_joint_warnings.py')
code=source.read_text().replace('from camera_joint_board import','from camera_factor_balanced_board import')
needle="  t=('net',name,None);n,k,pin=drivers[t]"
assert code.count(needle)==1
code=code.replace(needle,"""  matches=[t for t in drivers if t[1]+('['+str(t[2])+']' if t[2] is not None else '')==name]
  assert len(matches)==1,(name,matches)
  t=matches[0];n,k,pin=drivers[t]""")
start=code.index("  assert n.startswith('JOINT_LOCAL_')")
end=code.index(" result=dict(",start)
code=code[:start]+"""  assert k=='EFX_LUT4' and pin=='O' and loads[t]==0,(name,n,k,pin,loads[t])
  parent_graph=graph((PARENT/'outflow/ti60f225_oob.map.v').read_text()) if not explained else parent_graph
  assert n in parent_graph and parent_graph[n]==g[n],(n,'original endpoint changed')
  assert not re.search(r'^\\s*assign\\b.*?'+re.escape(name)+r'.*?;',source,re.M),(n,'top assign user')
  explained.append(dict(net=name,cell=n,reason='Original LUT endpoint now has zero consumers after CE reassignment; cell is unchanged and no top-level assign consumes it.'))
"""+code[end:]
exec(compile(code,str(source)+':factor-old-dead-endpoints','exec'),globals())
