"""Offline, manually annotated scene geometry target. NEVER an inference filter.

Whole architectural planes share vertices. Tree crowns use a common 3D voxel
lattice, perspective and face lighting; no luma-to-height mapping. This is an
annotated single-scene target prototype, not a general teacher model.
"""
from core import *

D=OUT/'r32-scene-geometry-teacher'

def polygon_mask(shape,points):
    m=np.zeros(shape[:2],np.uint8);cv2.fillPoly(m,[np.round(points).astype(np.int32)],1);return m.astype(bool)

def paint_plane(dst,src,points,factor=1.,colour=None):
    m=polygon_mask(src.shape,points)
    # Statistics of the whole material face, not colour-threshold components.
    c=np.median(src[m],axis=0) if colour is None else np.array(colour,float)
    dst[m]=np.clip(np.round(c*factor),0,255).astype(np.uint8)
    return m

def corridor(a):
    out=np.zeros_like(a);owner=np.zeros(a.shape[:2],np.uint16)
    # Coordinates are manual annotation of this source frame, not detected depth.
    faces=[
      ('ceiling',[(278,0),(786,0),(642,175),(495,175)],1.02),
      ('floor',[(204,720),(873,720),(645,348),(493,350)],1.0),
      ('left_wall',[(0,0),(278,0),(495,175),(493,350),(204,720),(0,720)],1.01),
      ('right_wall',[(786,0),(1280,0),(1280,720),(873,720),(645,348),(642,175)],1.0),
      ('back_wall',[(495,175),(642,175),(645,348),(493,350)],1.),
      ('near_right_column_front',[(782,0),(1009,0),(1002,720),(880,720),(782,525)],1.),
      ('near_right_column_side',[(1009,0),(1090,0),(1065,720),(1002,720)],.96),
      ('right_door',[(1090,0),(1280,0),(1265,720),(1065,720)],1.),
      ('left_near_column_front',[(0,0),(242,0),(283,601),(206,720),(0,720)],1.01),
      ('left_column_return',[(242,0),(278,0),(331,536),(283,601)],.95),
      ('left_middle_column',[(320,68),(350,81),(375,499),(331,536)],1.),
      ('left_middle_wall',[(350,81),(410,104),(427,422),(375,499)],1.02),
      ('left_far_column',[(410,104),(432,124),(449,395),(427,422)],1.),
      ('left_far_wall',[(432,124),(473,152),(473,372),(449,395)],1.),
      ('right_middle_return',[(740,49),(782,18),(782,525),(743,480)],.95),
      ('right_middle_wall',[(691,101),(740,49),(743,480),(699,408)],1.02),
      ('right_far_return',[(676,118),(691,101),(699,408),(681,385)],.98),
      ('right_far_wall',[(649,161),(676,118),(681,385),(648,352)],1.),
      ('back_door',[(504,183),(548,181),(558,351),(507,358)],1.),
      ('back_door_return',[(548,181),(557,184),(566,348),(558,351)],.84),
      ('left_skirt',[(204,720),(239,720),(503,364),(493,350)],1.),
      ('right_skirt',[(873,720),(905,720),(651,349),(645,348)],.9),
      ('back_skirt',[(507,358),(644,340),(645,351),(503,371)],.9),
    ]
    annotations=[]
    for i,(name,p,f) in enumerate(faces,1):
        mask=paint_plane(out,a,p,f);owner[mask]=i;annotations.append(dict(name=name,vertices=p))
    # Front and underside share the same beam vertices and remain continuous.
    beams=[([(302,0),(755,0),(717,36),(340,35)],[(340,35),(717,36),(711,50),(350,51)]),
           ([(395,69),(685,65),(674,89),(412,92)],[(412,92),(674,89),(668,103),(422,105)]),
           ([(447,128),(651,120),(646,144),(466,151)],[(466,151),(646,144),(642,157),(478,163)])]
    for j,(front,bottom) in enumerate(beams):
        for k,p in enumerate((front,bottom)):
            mask=paint_plane(out,a,p,1.02 if k==0 else .91);owner[mask]=100+j*2+k
    # Source-positioned large fixtures, not invented surface markings.
    for j,p in enumerate([
      [(91,48),(134,65),(143,143),(105,133)],
      [(248,0),(277,0),(312,22),(313,112),(253,89)],
      [(902,34),(935,23),(940,109),(905,120)],
      [(1187,162),(1255,143),(1256,303),(1188,315)],
      [(687,395),(716,384),(735,400),(728,477),(697,457)],
      [(477,353),(493,345),(508,365),(490,382)]
    ]):
        mask=paint_plane(out,a,p);owner[mask]=150+j
    # Floor tile lines follow the source plane; extremely low-contrast, optional.
    # No pixel-noise stipple, no bevel added to illumination or reflections.
    missing=owner==0
    out[missing]=a[missing]
    return out,owner,dict(kind='manual shared planar scene annotations',annotations=annotations,
                         unassigned_fraction=float(missing.mean()),new_geometry_is_manual_not_inferred=True)

def outdoor(a):
    h,w=a.shape[:2];out=a.copy();owner=np.zeros((h,w),np.uint16)
    # Architectural and ground planes share these manually identified boundaries.
    faces=[
      ('road',[(0,565),(201,493),(354,483),(664,473),(826,720),(0,720)],1.),
      ('right_paving',[(664,473),(1280,451),(1280,720),(826,720)],.98),
      ('left_house_front',[(0,315),(130,319),(142,455),(0,474)],1.),
      ('left_house_side',[(130,319),(217,334),(220,451),(142,455)],.96),
      ('left_house_roof_front',[(0,301),(128,310),(130,354),(0,350)],1.04),
      ('left_house_roof_side',[(128,310),(218,329),(220,367),(130,354)],.95),
      ('gate_left_column',[(302,374),(336,374),(337,440),(302,440)],1.),
      ('gate_beam_front',[(324,372),(552,360),(552,391),(337,398)],1.01),
      ('gate_right_column',[(533,360),(552,360),(555,444),(536,444)],1.),
      ('right_wall_front',[(736,344),(1022,324),(1034,439),(736,450)],1.),
      ('right_wall_coping',[(732,335),(1023,315),(1022,331),(736,351)],1.05),
    ]
    for i,(name,p,f) in enumerate(faces,1):
        m=paint_plane(out,a,p,f);owner[m]=i
    # Retain source road line: one fitted continuous strip, no source texture.
    paint_plane(out,a,[(0,688),(351,529),(355,527),(0,700)],colour=[201,175,108])
    # Entire canopy support is annotated. Texture is not promoted into depth.
    crown_regions=[[(0,0),(229,0),(289,112),(270,273),(216,346),(0,338)],
                   [(536,0),(1280,0),(1280,409),(1082,362),(953,328),(824,313),(699,318),(614,271),(563,156)]]
    support=np.zeros((h,w),bool)
    for p in crown_regions:support|=polygon_mask(a.shape,p)
    # Exclude sky/architecture inside the coarse crown annotations with a broad
    # manually scoped vegetation colour cue; it does not control cube heights.
    af=a.astype(float);green=(af[...,1]>af[...,0]*.91)&(af[...,1]>af[...,2]*1.06)&(af.mean(2)<200)
    vegetation=support&green
    vegetation=cv2.morphologyEx(vegetation.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((19,19),np.uint8)).astype(bool)
    smooth=cv2.GaussianBlur(a.astype(np.float32),(0,0),10)
    # Camera estimates shared by all 3D cuboids; physical world lattice.
    f=900.;cx=610.;cy=330.;camera_height=1.7
    def project(v):
        x,y,z=v;return [cx+f*x/z,cy+f*(camera_height-y)/z]
    # (image trunk x, image ground y, canopy centre height, radii, voxel size)
    trees=[(56,485,5.1,(2.5,2.0,2.0),.48),
           (228,441,5.7,(2.0,2.,2.),.48),
           (1210,566,4.8,(2.5,2.3,2.0),.42),
           (966,467,5.5,(2.2,2.,2.),.48),
           (844,454,5.2,(2.0,2.,2.),.48),
           (727,425,5.4,(1.8,2.,1.6),.48)]
    voxels={};trunks=[]
    for ti,(ix,iy,height,radii,s) in enumerate(trees):
        z=f*camera_height/(iy-cy);x=(ix-cx)*z/f
        trunks.append((x,z,height,s,ti))
        rx,ry,rz=radii
        for zz in np.arange(z-rz,z+rz,s):
          for yy in np.arange(height-ry,height+ry,s):
           for xx in np.arange(x-rx,x+rx,s):
            centre=np.array([xx+s/2,yy+s/2,zz+s/2])
            if ((centre[0]-x)/rx)**2+((centre[1]-height)/ry)**2+((centre[2]-z)/rz)**2>1:continue
            p=project(centre);px,py=np.round(p).astype(int)
            if -50<=px<w+50 and -50<=py<h+50:
                if vegetation[np.clip(py,0,h-1),np.clip(px,0,w-1)]:
                    # Index by global world origin, so touching blocks really meet.
                    q=np.floor(centre/s).astype(int)
                    voxels[(s,*q)]=ti
    primitives=[]
    def add_box(x0,y0,z0,sx,sy,sz,col,tag):
        v=[(x0,y0,z0),(x0+sx,y0,z0),(x0+sx,y0+sy,z0),(x0,y0+sy,z0),
           (x0,y0,z0+sz),(x0+sx,y0,z0+sz),(x0+sx,y0+sy,z0+sz),(x0,y0+sy,z0+sz)]
        faces=[([0,1,2,3],.87),([3,2,6,7],1.12),([0,4,5,1],.70),
               ([0,3,7,4],.75),([1,5,6,2],.98)]
        for ids,factor in faces:
            vv=np.array([v[k] for k in ids]);mid=vv.mean(0)
            # Show only physically camera-facing side/top/bottom planes.
            if ids==[3,2,6,7] and camera_height<y0+sy:continue
            if ids==[0,4,5,1] and camera_height>y0:continue
            if ids==[0,3,7,4] and x0<0:continue
            if ids==[1,5,6,2] and x0+sx>0:continue
            primitives.append((float(mid[2]),[project(p) for p in vv],np.clip(col*factor,0,255),tag))
    # Trunks are connected world-space cuboids, not separate bright colour layers.
    for x,z,height,s,ti in trunks:
        base=project((x,0,z));bx,by=np.round(base).astype(int)
        yy=np.clip(by-70,0,h-1);xx=np.clip(bx,0,w-1)
        col=smooth[yy,xx].astype(float)
        if col[1]>col[0]*1.25:col=np.array([87,71,48.])
        add_box(x-s*.45,0,z-s*.45,s*.9,height,s*.9,col,200+ti)
    for (s,qx,qy,qz),ti in voxels.items():
        x,y,z=np.array([qx,qy,qz])*s
        px,py=np.round(project((x+s/2,y+s/2,z+s/2))).astype(int)
        col=smooth[np.clip(py,0,h-1),np.clip(px,0,w-1)].astype(float)
        # Colour is a single material value for this complete cube.
        col[1]*=1.12;col[2]*=.90
        add_box(x,y,z,s,s,s,col,300+ti)
    # Erase the old leaf microtexture inside the support before projecting solids.
    # Covered holes retain material tone, never fake individual luma extrusions.
    out[vegetation]=np.clip(smooth[vegetation],0,255).astype(np.uint8)
    owner[vegetation]=90
    for depth,p,c,tag in sorted(primitives,key=lambda v:v[0],reverse=True):
        m=polygon_mask(a.shape,p);out[m]=c.round().astype(np.uint8);owner[m]=tag
    return out,owner,dict(kind='manual whole-scene planes plus world-space cuboid tree volumes',
        camera=dict(focal_px=f,principal_point=[cx,cy],height=camera_height),
        trees=trees,canopy_voxels=len(voxels),rendered_faces=len(primitives),
        target_only_manual_annotation=True,three_dimensional_geometry_is_estimated_not_measured=True,
        photo_retained_fraction=float((owner==0).mean()),unvoxelized_canopy_fraction=float((owner==90).mean()))

def main():
    setup();D.mkdir(exist_ok=True);disk_guard()
    report={'teacher_kind':'scene-specific manually annotated geometry prototype, not a trained general teacher',
            'B1_output':False,'training_started':False,'people_required':False,
            'source_script_sha256':sha(__file__),'scenes':{}}
    for key,path,fn in [('corridor',OUT/'data-quality/VID_20260924_231158_source.png',corridor),
                        ('outdoor',OUT/'targets/outdoor_source.png',outdoor)]:
        a=read(path);b,owner,info=fn(a)
        Image.fromarray(b).save(D/f'{key}_target.png');Image.fromarray(owner).save(D/f'{key}_surface_ids.png')
        board([[a,b]],D/f'{key}_comparison.jpg',['原图','场景几何教师原型 / 不是B1'],width=640)
        report['scenes'][key]=dict(source=str(path),source_sha256=sha(path),**info)
    save_json(D/'report.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
