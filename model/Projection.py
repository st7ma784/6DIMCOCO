import torch

def get_proj_fn(projection):
    projection=null
    if projection=="inv":
        projection=inv2
    elif projection=="iinv":
        projection=inv3
    elif projection=="None":
        projection=inv
def null(proj,im,text):
    return im,text
def inv(proj,im=None,text=None):
    return im,map(lambda text:text@proj,text)
def inv2(proj,im=[],text=[]):
    return map(lambda im: im@proj,im),text

def inv3(proj,im=[],text=[]):
    proj=torch.inverse(proj)
    return map(lambda im: im@proj,im),text
