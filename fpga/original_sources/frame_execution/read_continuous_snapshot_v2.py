"""Importable package entry point retaining all continuous read-only gates."""
from read_continuous_snapshot import checked_manifest,runner
runner.__file__=__file__
main=runner.main
if __name__=='__main__':main()
