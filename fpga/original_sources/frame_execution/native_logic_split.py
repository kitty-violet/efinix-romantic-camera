"""Linear-pass LOGIC COMB4 decomposition; retain exported primitive syntax."""
import re

INSTANCE = re.compile(r'^[ \t]*EFX_COMB4[ \t]+(\S+)[ \t]*\((.*?)\) /\*[^\n]*\*/[ \t]*;[^\n]*', re.M | re.S)
PARAMETER = re.compile(r'^[ \t]*defparam[ \t]+(\S+?)[ \t]*\.([A-Z_0-9]+)[ \t]*=[ \t]*([^;]+);', re.M)


def split_logic_comb(text):
    settings = {}
    for match in PARAMETER.finditer(text):
        name, key, value = match.groups()
        settings.setdefault(name, {})[key] = value.strip()
    rows = []
    def replace(match):
        name, body = match.groups()
        pins = dict(re.findall(r'\.(\w+)\(([^()]*)\)', body))
        params = settings[name]
        assert params['MODE'] == '"LOGIC"', (name, params)
        assert not any(pins.get(p, '').strip() for p in ('CO','FCO')), (name,pins)
        value = re.fullmatch(r"16'([hb])([0-9a-fA-F]+)", params['LUTMASK'])
        assert value, params
        mask = int(value[2], 16 if value[1]=='h' else 2)
        pmask = ((mask >> 8) & 15) * 0x1111
        lines = []
        generated = []
        for port, lut in (('O',mask), ('P',pmask)):
            if not pins.get(port, '').strip():
                continue
            instance = name + ('_splitO' if port == 'O' else '_splitP')
            if not instance.startswith('\\'):
                instance = '\\' + instance
            arguments = ', '.join('.'+p+'('+pins.get(p,"1'b0")+')' for p in ('I0','I1','I2','I3'))
            lines += [f"    EFX_LUT4 {instance} ({arguments}, .O({pins[port]})) /* verific EFX_ATTRIBUTE_CELL_NAME=EFX_LUT4, LUTMASK=16'h{lut:04x} */ ;",
                      f"    defparam {instance} .LUTMASK = 16'h{lut:04x};"]
            generated.append(instance.lstrip('\\'))
        rows.append(dict(name=name, mask=mask, p_mask=pmask, ports=pins, generated=generated))
        return '\n'.join(lines)
    changed = INSTANCE.sub(replace, text)
    removed_names = {row['name'] for row in rows}
    removed = []
    def remove_definition(match):
        if match[1] in removed_names:
            assert match[2] in ('LUTMASK','MODE'), match.group()
            removed.append((match[1],match[2]))
            return ''
        return match.group()
    changed = PARAMETER.sub(remove_definition, changed)
    assert len(removed) == 2 * len(rows)
    assert not INSTANCE.search(changed)
    return changed, rows
