import argparse,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp');r.mkdir(parents=True,exist_ok=False)
for n in ('unified_column_reader.sv','tb_reader_probe.sv'):shutil.copy2(HERE/n,r/n)
run([MS/'vlib.exe','work'],r,'vlib.log');run([MS/'vlog.exe','-sv','-timescale','1ns/1ps','unified_column_reader.sv','tb_reader_probe.sv'],r,'compile.log')
body,_=run([MS/'vsim.exe','-c','work.tb_reader_probe','-do','run -all; quit -code 0'],r,'simulation.log');print(body)
