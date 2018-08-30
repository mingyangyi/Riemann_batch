import torch

def norm(v):
    v = len(v.size())==2
    return v.norm(p=2, dim=1, keepdim=True)

def unit(v,eps=1e-8):
    vnorm = norm(v)
    return v/vnorm.add(eps)

def xTy(x,y):
    assert len(x.size()) == 2 and len(y.size()) == 2, 'xTy'
    return torch.sum(x * y, dim=1, keepdim=True)

import pdb
def clip_by_norm(v, clip_norm):
  v_norm = norm(v)
  if v.is_cuda:
    scale = torch.ones(v_norm.size()).cuda()
  else:
    scale = torch.ones(v_norm.size())
  mask = v_norm > clip_norm
  scale[mask] = clip_norm/v_norm[mask]

  return v*scale

def grassmann_project(x,eta):
    return eta-xTy(x,eta)*x

def grassmann_retrction(x,eta):
    return x * norm(eta).cos() + unit(eta) * norm(eta).sin()

def oblique_project(x,eta):
    ddiag=xTy(x,eta)
    p=eta-x*ddiag
    return p

def oblique_retrction(x,eta):
    v=x+eta
    return unit(v,1e-8)


