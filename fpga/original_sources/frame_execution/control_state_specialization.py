"""Independent, cycle-compatible control-state implementation experiment.

Preserve the released source and all bus/boot/reset behavior. The experiment
changes only the ten-state control register's encoding and CE inference.
The legacy four-bit state remains a diagnostic output, never a control input.
No claim that the board failure is caused by a particular synthesis primitive.
"""
import re

STATES = ('IDLE', 'DECODE', 'MAIL_READ', 'APB_SETUP', 'APB_ACCESS',
          'AXI_WRITE', 'AXI_B', 'AXI_AR', 'AXI_R', 'RESPONSE')


def specialize(source):
    start = source.index('module cnn_fixed_control_soc (')
    before, control = source[:start], source[start:]
    assert control.count('endmodule') == 1
    old = (' localparam IDLE=0,DECODE=1,MAIL_READ=2,APB_SETUP=3,APB_ACCESS=4,\n'
           '  AXI_WRITE=5,AXI_B=6,AXI_AR=7,AXI_R=8,RESPONSE=9;\n'
           ' reg [3:0] state;')
    assert control.count(old) == 1
    end = control.index(' // Read-only, configuration-initialized evidence')
    functional, observation = control[:end], control[end:]
    # Exact old declarations are removed before renaming their uses.
    functional = functional.replace(old, ' __CONTROL_STATE_DECLARATIONS__')
    functional = re.sub(r'\bstate\b', 'state_oh', functional)
    declarations = (' // Fixed one-hot control; keep legacy diagnostics separately.\n'
        ' localparam [9:0] ' + ','.join(f"{n}=10'h{1 << i:03x}" for i,n in enumerate(STATES)) + ';\n'
        ' (* syn_preserve="true", syn_extract_enable="false" *) reg [9:0] state_oh;\n'
        ' wire [3:0] state;\n' + '\n'.join(
            f' assign state[{bit}]=' + ' | '.join(f'state_oh[{i}]' for i in range(10) if (i >> bit) & 1) + ';'
            for bit in range(4)))
    functional = functional.replace(' __CONTROL_STATE_DECLARATIONS__', declarations)
    assert 'case(state_oh)' in functional and 'case(state)' not in functional
    return before + functional + observation


def audit_delta(original, candidate):
    assert candidate == specialize(original)
    return dict(scope='Control state encoding and CE inference only',
                bus_protocol_unchanged=True, reset_interface_unchanged=True,
                observation_encoding_unchanged_for_legal_states=True,
                physical_root_cause_proven=False, hardware_access=False)
