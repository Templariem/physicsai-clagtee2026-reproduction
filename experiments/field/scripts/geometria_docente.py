"""Camera registration matching the ruler application, using semantic masks only."""
import cv2,numpy as np
from fem_regla import Scene,SPHERE

def contour(mask,cls):
    cc,_=cv2.findContours((mask==cls).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cc=[c[:,0].astype('float32') for c in cc if cv2.contourArea(c)>=16]
    return max(cc,key=cv2.contourArea) if cc else None

def scene_from_mask(mask,domain_scale=16.):
    polys=[contour(mask,c) for c in [1,2,3]]
    if polys[0] is None or polys[1] is None:raise ValueError('missing sphere or coil')
    centers=[]
    for poly in polys[:2]:
        m=cv2.moments(poly);centers.append(np.array([m['m10']/m['m00'],m['m01']/m['m00']]))
    origin,coil=centers;radius=float(np.mean(np.linalg.norm(polys[0]-origin,axis=1)))
    if radius<3 or coil[1]<=origin[1] or abs(coil[0]-origin[0])>=3*radius or coil[1]-origin[1]>=12*radius:raise ValueError('invalid source pairing')
    up=origin-coil;up/=np.linalg.norm(up);basis=np.stack([[-up[1],up[0]],up],axis=1);scale=.045/(2*radius)
    transform=(origin,basis,scale)
    if polys[2] is None:return Scene(mode='none',domain_scale=domain_scale),transform
    box=cv2.boxPoints(cv2.minAreaRect(polys[2]));p=(box-origin)@basis*scale+SPHERE
    edges=np.roll(p,-1,axis=0)-p;lengths=np.linalg.norm(edges,axis=1);axis=edges[np.argmax(lengths)];axis/=np.linalg.norm(axis)
    midpoint=p.mean(0)
    if np.dot(axis,midpoint-SPHERE)<0:axis=-axis
    center=midpoint+(.085-lengths.max()/2)*axis
    scene=Scene(x_cm=round(float(center[0]*100)/.2)*.2,y_cm=round(float(center[1]*100)/.2)*.2,
                angle_deg=round(float(np.degrees(np.arctan2(axis[1],axis[0])))/2)*2,mode='ground',domain_scale=domain_scale)
    return scene.checked(),transform

def world_to_uv(points,transform):
    origin,basis,scale=transform
    return (((points-SPHERE)/scale)@basis.T+origin)/447
