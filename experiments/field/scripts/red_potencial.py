"""Image tokens + RGB -> dense LED-equivalent voltage. No mask/geometry inputs."""
import torch
from torch import nn
from torch.nn import functional as F
def block(a,b,stride=1):return nn.Sequential(nn.Conv2d(a,b,3,stride=stride,padding=1,bias=False),nn.GroupNorm(8,b),nn.GELU())
class Potential(nn.Module):
    def __init__(self,dimension,voltage_lut):
        super().__init__();self.project=nn.Sequential(nn.Conv2d(dimension,64,1),nn.GroupNorm(8,64),nn.GELU())
        self.rgb2=block(3,16,2);self.rgb4=block(16,32,2);self.fuse=nn.Sequential(block(96,64),block(64,64))
        self.up2=block(80,32);self.up1=block(32,16);self.logits=nn.Conv2d(16,1,1)
        self.register_buffer('voltage_lut',torch.as_tensor(voltage_lut,dtype=torch.float32))
    def calibrate(self,phi):
        p=phi.float().clamp(0,1)*(len(self.voltage_lut)-1);lo=p.long().clamp(0,len(self.voltage_lut)-2);a=p-lo
        lut=self.voltage_lut.float();return lut[lo]*(1-a)+lut[lo+1]*a
    def forward(self,tokens,rgb,return_phi=False):
        r2=self.rgb2(rgb);r4=self.rgb4(r2);x=F.interpolate(self.project(tokens),r4.shape[-2:],mode='bilinear',align_corners=False)
        x=self.fuse(torch.cat([x,r4],1));x=F.interpolate(x,r2.shape[-2:],mode='bilinear',align_corners=False)
        x=self.up2(torch.cat([x,r2],1));phi=self.logits(self.up1(F.interpolate(x,rgb.shape[-2:],mode='bilinear',align_corners=False))).float().sigmoid()
        voltage=self.calibrate(phi)
        return (voltage,phi) if return_phi else voltage

def loss(model,voltage,phi,target,target0,valid,air,rule):
    truth=model.calibrate(target);before=model.calibrate(target0);effect=((before-truth).abs()>.1)&air.bool()
    weights=valid*(1+4*effect.float()+9*rule.float())
    fit=(F.smooth_l1_loss(voltage/3,truth/3,reduction='none',beta=.1)*weights).sum()/weights.sum().clamp_min(1)
    normalized=((phi-target).square()*weights).sum()/weights.sum().clamp_min(1)
    gradient=phi.new_zeros(())
    for dim in [-1,-2]:
        a=phi.diff(dim=dim);b=target.diff(dim=dim);v=valid.narrow(dim,0,valid.shape[dim]-1)*valid.narrow(dim,1,valid.shape[dim]-1)
        gradient+=((a-b).abs()*v).sum()/v.sum().clamp_min(1)
    return fit+.2*normalized+gradient
