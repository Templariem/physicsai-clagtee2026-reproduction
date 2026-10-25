"""Coordinate surrogate of the frozen v2 FEM. No visual encoder or RGB input."""
import torch
from torch import nn

def features(raw):
    q=raw[:,:2];center=raw[:,2:4];cs=raw[:,4:6];present=raw[:,9:10]
    base=raw[:,6:7]
    delta=q-center;u=(delta*cs).sum(1,keepdim=True)
    v=-delta[:,0:1]*cs[:,1:2]+delta[:,1:2]*cs[:,0:1]
    d=torch.cat([u.abs()-.085,v.abs()-.0085],1)
    dr=torch.linalg.vector_norm(torch.clamp(d,min=0)+1e-10,dim=1,keepdim=True)+torch.clamp(d.max(1,keepdim=True).values,max=0)
    sphere=torch.linalg.vector_norm(q-q.new_tensor([0.,.1225])+1e-10,dim=1,keepdim=True)-.0225
    dc=(q-q.new_tensor([0.,.052])).abs()-q.new_tensor([.015,.052])
    coil=torch.linalg.vector_norm(torch.clamp(dc,min=0)+1e-10,dim=1,keepdim=True)+torch.clamp(dc.max(1,keepdim=True).values,max=0)
    ds=torch.minimum(sphere,coil)
    # Actual domain16 boundaries, never the edge of a camera image.
    margin=torch.cat([q[:,0:1]+5.6,5.6-q[:,0:1],q[:,1:2]+4.205,4.595-q[:,1:2]],1).min(1,keepdim=True).values
    ps=torch.clamp(ds,min=0);pr=torch.clamp(dr,min=0)
    w=torch.exp(-pr/.08)*ps/(ps+pr+1e-9)*present
    lift=base*(1-w);envelope=torch.clamp(margin/.025,0,1)*present
    f=torch.cat([q[:,0:1]/.35,(q[:,1:2]-.195)/.275,center[:,0:1]/.35,(center[:,1:2]-.195)/.275,
                 cs[:,0:1]**2-cs[:,1:2]**2,2*cs[:,0:1]*cs[:,1:2],
                 torch.clamp(u.abs()/.17,max=40),torch.clamp(v.abs()/.05,max=120),
                 torch.clamp(sphere/.1,-1,60),torch.clamp(coil/.1,-1,60),torch.clamp(dr/.1,-1,60),
                 base,raw[:,7:9]*.05,w,present],1)
    return f,lift,envelope,ds,dr,margin

class PhysicalField(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(16,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,128),nn.SiLU(),nn.Linear(128,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,raw):
        f,lift,envelope,ds,dr,margin=features(raw)
        out=torch.sigmoid(torch.logit(lift.clamp(1e-6,1-1e-6))+envelope*4*torch.tanh(self.net(f)))
        out=torch.where(raw[:,9:10]>0,out,raw[:,6:7])
        out=torch.where(ds<=0,torch.ones_like(out),out)
        out=torch.where((dr<=0)&(raw[:,9:10]>0),torch.zeros_like(out),out)
        return torch.where(margin<=0,torch.zeros_like(out),out)
