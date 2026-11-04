"""Rebind the unchanged controller checks to the cached-head full-board snapshot."""
from cached_head_flow import bind

if __name__ == '__main__':
    bind()
    from validate_local_control_decode import main
    main()
