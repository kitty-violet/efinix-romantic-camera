"""Controlled public-wrapper edit: remove three unconnected peripherals.

CPU/JTAG/OCR/CLINT/APB/AXI and encrypted CPU implementations stay intact.
The existing BMB decoder's no-hit path rejects the removed address windows.
"""
import re


def prune(text):
    for module, instance in [('BmbUartCtrl', 'system_uart_0_io_logic'),
                             ('BmbSpiXdrMasterCtrl', 'system_spi_0_io_logic'),
                             ('BmbI2cCtrl', 'system_i2c_0_io_logic')]:
        pattern = rf'  {module} {instance} \(.*?\n  \);'
        matches = list(re.finditer(pattern, text, re.S))
        assert len(matches) == 1, instance
        old = matches[0].group()
        outputs = re.findall(r'\.([\w]+)\s*\(([^)]+)\)\s*,?\s*//o', old)
        assert outputs
        lines = [f'  // {instance}: physically unconnected; address window is unmapped below.']
        for port, signal in outputs:
            signal = signal.strip()
            # No command can reach these removed leaves. They must never
            # inject unsolicited responses or interrupts into the live bus.
            high = port.endswith('cmd_ready') or port.endswith('payload_last') or port in (
                'io_uart_txd', 'io_i2c_sda_write', 'io_i2c_scl_write', 'io_spi_ss')
            lines.append(f"  assign {signal} = 'b{int(high)};")
        text = text.replace(old, '\n'.join(lines))
    for index, mask, address in [(2, '00003f', '001000'), (3, '000fff', '006000'),
                                 (4, '0000ff', '00a000')]:
        old = f"  assign logic_hitsS0_{index} = ((io_input_cmd_payload_fragment_address & (~ 24'h{mask})) == 24'h{address});"
        assert text.count(old) == 1
        text = text.replace(old, f"  assign logic_hitsS0_{index} = 1'b0; // Removed peripheral: existing no-hit ERROR response.")
    return text
