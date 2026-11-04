"""Colour codes with monotone major layers and unit-weight remainders.

The6/5/4 layout keeps the same15/9/9 totals in15 binary fields. This is a
training/weight-construction helper, never a deployed lookup operation.
"""
import torch

RADIX_WEIGHTS=((1,1,3,3,3,3),(1,1,2,2,2),(1,1,3,3))
RADIX_BASES=(3,2,3)


def radix_tables(device='cpu'):
    result=[]
    for weights,base in zip(RADIX_WEIGHTS,RADIX_BASES):
        values=torch.arange(sum(weights)+1,device=device);major_count=len(weights)-2
        quotient=(values//base).clamp_max(major_count);remainder=values-base*quotient
        assert remainder.max()<=2
        bits=torch.cat(((remainder[:,None]>torch.arange(2,device=device)[None]),
                        (quotient[:,None]>torch.arange(major_count,device=device)[None])),1).float()
        assert torch.equal((bits*bits.new_tensor(weights)).sum(1),values.float())
        assert (torch.diff(bits[:,2:],dim=0)>=0).all()
        result.append(bits)
    return result


def radix_fields(code):
    return torch.cat([table[level] for table,level in zip(radix_tables(code.device),(code//81,code//9%9,code%9))],-1).permute(0,3,1,2)


def fields_for_layout(code,setup):
    if setup.get('code_layout')=='radix_thermometer_654':return radix_fields(code)
    from .probe_b1_symmetric_binary_coverage import code_fields
    return code_fields(code)
