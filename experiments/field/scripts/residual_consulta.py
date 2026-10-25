"""Declared query extension of historical frozen pooled PCA128 residual MLP.
Geometry input is privileged. This is not an unchanged historical scalar model.
"""
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from modelo import Model as DenseModel
from red_potencial import Potential
from red_fisica import features
ROOT=Path(__file__).resolve().parents[1]
class ResidualPotential(nn.Module):
 def __init__(self,kind,lut):
  super().__init__();p=np.load(ROOT/'modelos'/f'pca_{kind}_train.npz')
  for k in ['mean','components','scale']:self.register_buffer('pca_'+k,torch.tensor(p[k],dtype=torch.float32))
  self.register_buffer('voltage_lut',torch.as_tensor(lut,dtype=torch.float32));self.context=None
  self.net=nn.Sequential(nn.Linear(144,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.2),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 calibrate=Potential.calibrate
 def forward(self,tokens,rgb,return_phi=False):
  assert self.context is not None and self.context.shape[1]==11
  raw=self.context[:,:10].float();base=self.context[:,10:11].float();b=raw.shape[0]
  pooled=tokens.float().mean((2,3));v=((pooled-self.pca_mean)@self.pca_components.T)/self.pca_scale
  query=F.interpolate(raw,(112,112),mode='bilinear',align_corners=False).permute(0,2,3,1).reshape(-1,10)
  with torch.no_grad():f=features(query)[0]
  v=v[:,None,None,:].expand(-1,112,112,-1).reshape(-1,128)
  delta=self.net(torch.cat([v,f],1)).reshape(b,1,112,112).float().tanh()*.25
  phi=(base+F.interpolate(delta,(448,448),mode='bilinear',align_corners=False)).clamp(0,1)
  # Restore conductor and outer-domain values; identical geometry assumptions to baseline.
  qx,qy=raw[:,0:1],raw[:,1:2];dx,dy=qx-raw[:,2:3],qy-raw[:,3:4];cs,sn=raw[:,4:5],raw[:,5:6]
  source=(qx.square()+(qy-.1225).square()<=.0225**2)|((qx.abs()<=.015)&(qy>=0)&(qy<=.104))
  ruler=((dx*cs+dy*sn).abs()<=.085)&((-dx*sn+dy*cs).abs()<=.0085)&(raw[:,9:10]>.5)
  phi=torch.where(source,torch.ones_like(phi),phi);phi=torch.where(ruler,torch.zeros_like(phi),phi)
  outside=(qx<=-5.6)|(qx>=5.6)|(qy<=-4.205)|(qy>=4.595);phi=torch.where(outside,torch.zeros_like(phi),phi)
  voltage=self.calibrate(phi);return (voltage,phi) if return_phi else voltage
class ResidualModel(DenseModel):
 def __init__(self,kind,lut):
  super().__init__(kind,'frozen',lut,False);self.arm='residual'
  with torch.random.fork_rng(devices=[]):torch.manual_seed(4242);self.head=ResidualPotential(kind,lut)
 def train(self,mode=True):
  super().train(mode);self.encoder.eval();return self
 def checkpoint(self):
  ck=super().checkpoint();ck.update(architecture='query_extended_legacy_PCA128_MLP64_32_gate025',geometry_input=True,physical_baseline='v3 retrained physical ensemble');return ck
