"""Word-sized diagnostic mailbox, original proven observation transport."""
from pathlib import Path
original=Path(__file__).with_name('prepare_video_service_native.py')
code=original.read_text()
changes=[
 ('from video_service_display_native import extension as display_extension,explicit_net_assignments,split_net_declarations',
  'from video_service_word_native import extension as display_extension,explicit_net_assignments,split_net_declarations'),
 ("[SOURCE,Path(__file__),HERE/'video_service_display_native.py'",
  "[SOURCE,Path(__file__),HERE/'prepare_video_service_native.py',HERE/'video_service_word_native.py',HERE/'variants/video_service_word/video_debug_mailbox.sv',HERE/'video_service_display_native.py'"),
 ("(out/filename).write_bytes((VARIANT/filename).read_bytes())",
  "(out/filename).write_bytes((HERE/'variants/video_service_word'/filename if filename=='video_debug_mailbox.sv' else VARIANT/filename).read_bytes())")]
for old,new in changes:
 assert code.count(old)==1,old
 code=code.replace(old,new)
exec(compile(code,str(original)+':word-original-transport','exec'),globals())
