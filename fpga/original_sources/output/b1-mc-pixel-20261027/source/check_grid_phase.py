"""Audit strict6x6-grid representability; separate from visual block size."""
from core import *
from make_large_pixel_targets import six_target

@torch.inference_mode()
def main():
    setup();rng=np.random.default_rng(271027)
    a=rng.integers(0,256,(128,128,3),dtype=np.uint8)
    # A smooth oblique boundary is a clearer lattice-phase probe than white noise.
    yy,xx=np.mgrid[:128,:128];a=np.where((xx+yy*.7>99)[...,None],np.array([220,183,80]),np.array([27,57,145])).astype(np.uint8)
    shift=lambda z:np.pad(z,((0,0),(4,0),(0,0)),mode='edge')[:,:128]
    b=shift(a);crop=(slice(44,84),slice(44,84))
    cases=[]
    for name,p in [('pixel_v1',OUT/'exports/mc-pixel-v1/coefficients-padded.bin'),('six_target_fit',OUT/'flat-pixel-v2/candidate.bin')]:
        net=IntegerEngine(p.read_bytes());ya=rgb(net(tensor(a)[None]*128)/128);yb=rgb(net(tensor(b)[None]*128)/128)
        e=np.abs(yb[crop].astype(int)-shift(ya)[crop].astype(int))
        assert e.max()==0
        cases.append(dict(model=name,shift_pixels=4,interior_max_error=int(e.max()),coefficient_sha256=sha(p)))
    ta=six_target(a);tb=six_target(b);error=np.abs(tb[crop].astype(float)-shift(ta)[crop])
    assert error.max()>0
    save_json(OUT/'flat-pixel-v2/grid_phase_audit.json',dict(students=cases,
        strict6_teacher_under4_shift=dict(interior_mae=float(error.mean()),max_error=float(error.max())),
        inference='Fixed convolution/stride2/stride2/nearest2/nearest2 graph is translation-equivariant in4-pixel interior steps. A global6-grid teacher is not. Exact global6 cells for arbitrary images cannot be achieved by coefficients alone.',
        limitation='Does not prohibit content-dependent larger colour regions or approximate6-scale appearance.',
        changed_topology=False))
    print('GRID_PHASE_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
