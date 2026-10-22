# GDR-Net upstream rotation utilities, Apache-2.0; see GDRNET_LICENSE.txt
import torch
from torch.nn import functional as F

def normalize_vector(v):
    # bxn
    # batch = v.shape[0]
    # v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    # v_mag = torch.max(v_mag, torch.FloatTensor([1e-8]).to(v))
    # v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    # v = v / v_mag
    v = F.normalize(v, p=2, dim=1)
    return v


def cross_product(u, v):
    # u, v bxn
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # bx3

    return out


def ortho6d_to_mat_batch(poses):
    # poses bx6
    # poses
    x_raw = poses[:, 0:3]  # bx3
    y_raw = poses[:, 3:6]  # bx3

    x = normalize_vector(x_raw)  # bx3
    z = cross_product(x, y_raw)  # bx3
    z = normalize_vector(z)  # bx3
    y = cross_product(z, x)  # bx3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # bx3x3
    return matrix

