"""RGB -> potential; no masks, geometry, detector or solver in forward."""
from pathlib import Path
import contextlib,json,os,hashlib,math
import torch
from torch import nn
from transformers import AutoModel,IJepaModel
from red_potencial import Potential
ROOT=Path(__file__).resolve().parents[1]
ENCODERS=Path(os.environ.get('PHYSICS_ENCODERS',str(ROOT.parents[1]/'encoders')))

class Model(nn.Module):
    def __init__(self,kind,arm,lut,checkpointing=True):
        super().__init__();assert kind in ['dinov3','ijepa'] and arm in ['frozen','partial','full']
        self.kind,self.arm=kind,arm
        cls=AutoModel if kind=='dinov3' else IJepaModel
        self.encoder=cls.from_pretrained(ENCODERS/kind,local_files_only=True,attn_implementation='sdpa')
        self.encoder.requires_grad_(arm=='full')
        blocks=[(name,module) for name,module in self.encoder.named_modules() if isinstance(module,nn.ModuleList) and len(module)==self.encoder.config.num_hidden_layers]
        assert len(blocks)==1,[(n,len(m)) for n,m in blocks]
        block_name,layers=blocks[0];self.partial_blocks=math.ceil(len(layers)/4)
        if arm=='partial':
            for layer in layers[-self.partial_blocks:]:layer.requires_grad_(True)
            for name in ['norm','layernorm']:
                if hasattr(self.encoder,name):getattr(self.encoder,name).requires_grad_(True)
        if arm!='frozen' and checkpointing:self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.dimension=self.encoder.config.hidden_size;self.grid=448//self.encoder.config.patch_size
        # Reset after encoder loading, so all arms start from identical decoder weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4242);self.head=Potential(self.dimension,lut)
        cfg=json.loads((ROOT/'config/encoders.json').read_text())[kind]['processor']
        self.register_buffer('mean',torch.tensor(cfg['image_mean']).view(1,3,1,1),persistent=False)
        self.register_buffer('std',torch.tensor(cfg['image_std']).view(1,3,1,1),persistent=False)
        self.trainable_encoder_names=[n for n,p in self.encoder.named_parameters() if p.requires_grad]
        self.trainable_encoder_params=sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        self.block_path=block_name
    def train(self,mode=True):
        super().train(mode)
        if self.arm=='frozen':self.encoder.eval()
        return self
    def forward(self,rgb,return_phi=False):
        context=torch.no_grad() if self.arm=='frozen' else contextlib.nullcontext()
        with context:
            out=self.encoder(pixel_values=(rgb-self.mean)/self.std,interpolate_pos_encoding=True)
            tokens=out.last_hidden_state
            if self.kind=='dinov3':tokens=tokens[:,1+self.encoder.config.num_register_tokens:]
            assert tokens.shape[1]==self.grid**2,tokens.shape
            tokens=tokens.reshape(rgb.shape[0],self.grid,self.grid,self.dimension).permute(0,3,1,2)
        return self.head(tokens,rgb,return_phi)
    def checkpoint(self):
        trainable=set(self.trainable_encoder_names)
        return dict(kind=self.kind,arm=self.arm,dimension=self.dimension,
            head={k:v.detach().cpu() for k,v in self.head.state_dict().items()},
            encoder_delta={k:v.detach().cpu() for k,v in self.encoder.state_dict().items() if k in trainable},
            trainable_encoder_names=self.trainable_encoder_names,partial_blocks=self.partial_blocks,
            output='dense LED-equivalent voltage [B,1,448,448]',yolo_inference=False,mask_inference=False,FEM_inference=False)
    def restore(self,ck):
        assert ck['kind']==self.kind and ck['arm']==self.arm
        self.head.load_state_dict(ck['head']);result=self.encoder.load_state_dict(ck['encoder_delta'],strict=False)
        assert not result.unexpected_keys
        assert set(ck['encoder_delta'])==set(self.trainable_encoder_names)
