"""6x6 flat-pixel route labels, kept separate from MC shape studies."""
from pixel_targets import *

def six_target(a):
    h,w=a.shape[:2];ph=(-h)%6;pw=(-w)%6
    z=np.pad(a,((0,ph),(0,pw),(0,0)),mode='edge')
    z=z.astype(np.float32).reshape(z.shape[0]//6,6,z.shape[1]//6,6,3).mean((1,3))
    q=np.zeros_like(z)
    for t in (np.arange(5)+.5)*51:
        q+=51*np.clip((z-t)/14.+.5,0,1)
    # Preserve v1 colour-block mapping; apply only a gentle final hue-preserving boost.
    y=q@np.array([.2126,.7152,.0722],np.float32)
    z=y[...,None]+1.03*(q-y[...,None])
    z=np.clip(z/255,0,1)**.97*255
    assert np.isfinite(z).all()
    return np.clip(z+.5,0,255).astype(np.uint8).repeat(6,0).repeat(6,1)[:h,:w]

def main():
    d=ROOT/'flat-pixel-v2/targets';d.mkdir(parents=True,exist_ok=True)
    for s in ['outdoor','indoor']:
        a=np.asarray(Image.open(ROOT/f'targets/{s}_source.png').convert('RGB'))
        old=np.asarray(Image.open(ROOT/f'evaluation/{s}_student.png').convert('RGB'))
        q=six_target(a);Image.fromarray(q).save(d/f'{s}_6x6_target.png')
        panel([old,q],['当前4×4学生 / 纯B1整数','放大到6×6的目标 / 非学生'],d/f'{s}_comparison.jpg')
        crop=(slice(192,384),slice(448,768))
        panel([old[crop],q[crop]],['当前学生局部 ×2','6×6目标局部 ×2'],d/f'{s}_detail.png')
    (d.parent/'config.json').write_text(json.dumps(dict(route='flat pixel enlargement, distinct from MC geometry',
        cell=6,saturation=1.03,gamma=.97,levels=6,transition_rgb=14,teacher_only=True,student_exported=False,
        exact_grid_note='Fixed B1 stride total4 is equivariant under4-pixel translations; global6-pixel lattice is not. Must audit learned approximation instead of claiming exact6.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('LARGE_PIXEL_TARGETS_COMPLETE',flush=True)

if __name__=='__main__':main()
