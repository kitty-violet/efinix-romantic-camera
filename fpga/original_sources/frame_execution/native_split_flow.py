"""Bind existing OFP5 mapped gates to the native-map lineage, offline only."""
from native_split_board import BASE,audit_delta


def main():
    import startup_cause_flow as flow
    flow.BASE=BASE;flow.audit_delta=audit_delta
    flow.main()


if __name__=='__main__':main()
