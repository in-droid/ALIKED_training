import cv2
import torch
import numpy as np

from copy import deepcopy



def compute_keypoints_distance(kpts0, kpts1, p=2):
    """
    Args:
        kpts0: torch.tensor [M,2]
        kpts1: torch.tensor [N,2]
        p: (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm

    Returns:
        dist, torch.tensor [N,M]
    """
    dist = kpts0[:, None, :] - kpts1[None, :, :]  # [M,N,2]
    dist = torch.norm(dist, p=p, dim=2)  # [M,N]
    return dist


def keypoints_normal2pixel(kpts_normal, w, h):
    wh = kpts_normal[0].new_tensor([[w - 1, h - 1]])
    kpts_pixel = [(kpts + 1) / 2 * wh for kpts in kpts_normal]
    return kpts_pixel




def mutual_argmax(value, mask=None, as_tuple=True):
    """
    Args:
        value: MxN
        mask:  MxN

    Returns:

    """
    value = value - value.min()  # convert to non-negative tensor
    if mask is not None:
        value = value * mask

    max0 = value.max(dim=1, keepdim=True)  # the col index the max value in each row
    max1 = value.max(dim=0, keepdim=True)

    valid_max0 = value == max0[0]
    valid_max1 = value == max1[0]

    mutual = valid_max0 * valid_max1
    if mask is not None:
        mutual = mutual * mask

    return mutual.nonzero(as_tuple=as_tuple)


def mutual_argmin(value, mask=None):
    return mutual_argmax(-value, mask)


def project(points3d, K):
    """
    project 3D points to image plane

    Args:
        points3d: [N,3]
        K: [3,3]

    Returns:
        uv, (u,v), [N,2]
    """
    if type(K) == torch.Tensor:
        zuv1 = torch.einsum('jk,nk->nj', K, points3d)  # z*(u,v,1) = K*points3d -> [N,3]
    elif type(K) == np.ndarray:
        zuv1 = np.einsum('jk,nk->nj', K, points3d)
    else:
        raise TypeError("Input type should be 'torch.tensor' or 'numpy.ndarray'")
    uv1 = zuv1 / zuv1[:, -1][:, None]  # (u,v,1) -> [N,3]
    uv = uv1[:, 0:2]  # (u,v) -> [N,2]
    return uv, zuv1[:, -1]



def unproject(uv, d, K):
    """
    unproject pixels uv to 3D points

    Args:
        uv: [N,2]
        d: depth, [N,1]
        K: [3,3]

    Returns:
        3D points, [N,3]
    """
    duv = uv * d  # (u,v) [N,2]
    if type(K) == torch.Tensor:
        duv1 = torch.cat([duv, d], dim=1)  # z*(u,v,1) [N,3]
        K_inv = torch.inverse(K)  # [3,3]
        points3d = torch.einsum('jk,nk->nj', K_inv, duv1)  # [N,3]
    elif type(K) == np.ndarray:
        duv1 = np.concatenate((duv, d), axis=1)  # z*(u,v,1) [N,3]
        K_inv = np.linalg.inv(K)  # [3,3]
        points3d = np.einsum('jk,nk->nj', K_inv, duv1)  # [N,3]
    else:
        raise TypeError("Input type should be 'torch.tensor' or 'numpy.ndarray'")
    return points3d



def plot_keypoints(image, kpts, radius=2, color=(255, 0, 0)):
    image = image.cpu().detach().numpy() if isinstance(image, torch.Tensor) else image
    kpts = kpts.cpu().detach().numpy() if isinstance(kpts, torch.Tensor) else kpts

    if image.dtype is not np.dtype('uint8'):
        image = image * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    out = np.ascontiguousarray(deepcopy(image))
    kpts = np.round(kpts).astype(int)

    for kpt in kpts:
        y0, x0 = kpt
        cv2.drawMarker(out, (x0, y0), color, cv2.MARKER_CROSS, radius)

        # cv2.circle(out, (x0, y0), radius, color, -1, lineType=cv2.LINE_4)
    return out


def to_homogeneous(kpts):
    '''
    :param kpts: Nx2
    :return: Nx3
    '''
    ones = kpts.new_ones([kpts.shape[0], 1])
    return torch.cat((kpts, ones), dim=1)





def warp_homography(kpts0, params):
    '''
    :param kpts: Nx2
    :param homography_matrix: 3x3
    :return:
    '''
    homography_matrix = params['homography']
    w, h = params['width'], params['height']
    kpts0_homogeneous = to_homogeneous(kpts0)
    kpts01_homogeneous = torch.einsum('ij,kj->ki', homography_matrix, kpts0_homogeneous)
    kpts01 = kpts01_homogeneous[:, :2] / kpts01_homogeneous[:, 2:]

    kpts01_ = kpts01.detach()
    # due to float coordinates, the upper boundary should be (w-1) and (h-1).
    # For example, if the image size is 480, then the coordinates should in [0~470].
    # 470.5 is not acceptable.
    valid01 = (kpts01_[:, 0] >= 0) * (kpts01_[:, 0] <= w - 1) * (kpts01_[:, 1] >= 0) * (kpts01_[:, 1] <= h - 1)
    kpts0_valid = kpts0[valid01]
    kpts01_valid = kpts01[valid01]
    ids = torch.nonzero(valid01, as_tuple=False)[:, 0]
    ids_out = torch.nonzero(~valid01, as_tuple=False)[:, 0]

    # kpts0_valid: valid keypoints0, the invalid and inconsistance keypoints are removed
    # kpts01_valid: the warped valid keypoints0
    # ids: the valid indices
    return kpts0_valid, kpts01_valid, ids, ids_out



def warp_se3(kpts0, params):
    pose01 = params['pose01']  # relative motion
    bbox0 = params['bbox0']  # row, col
    bbox1 = params['bbox1']
    depth0 = params['depth0']
    depth1 = params['depth1']
    intrinsics0 = params['intrinsics0']
    intrinsics1 = params['intrinsics1']

    # kpts0_valid: valid kpts0
    # z0_valid: depth of valid kpts0
    # ids0: the indices of valid kpts0 ( valid corners and valid depth)
    # ids0_valid_corners: the valid indices of kpts0 in image ( 0<=x<w, 0<=y<h )
    # ids0_valid_depth: the valid indices of kpts0 with valid depth ( depth > 0 )
    z0_valid, kpts0_valid, ids0, ids0_valid_corners, ids0_valid_depth = interpolate_depth(kpts0, depth0)

    # COLMAP convention
    bkpts0_valid = kpts0_valid + bbox0[[1, 0]][None, :] + 0.5

    # unproject pixel coordinate to 3D points (camera coordinate system)
    bpoints3d0 = unproject(bkpts0_valid, z0_valid.unsqueeze(1), intrinsics0)  # [:,3]
    bpoints3d0_homo = to_homogeneous(bpoints3d0)  # [:,4]

    # warp 3D point (camera 0 coordinate system) to 3D point (camera 1 coordinate system)
    bpoints3d01_homo = torch.einsum('jk,nk->nj', pose01, bpoints3d0_homo)  # [:,4]
    bpoints3d01 = bpoints3d01_homo[:, 0:3]  # [:,3]

    # project 3D point (camera coordinate system) to pixel coordinate
    buv01, z01 = project(bpoints3d01, intrinsics1)  # uv: [:,2], (h,w); z1: [N]

    uv01 = buv01 - bbox1[None, [1, 0]] - .5

    # kpts01_valid: valid kpts01
    # z01_valid: depth of valid kpts01
    # ids01: the indices of valid kpts01 ( valid corners and valid depth)
    # ids01_valid_corners: the valid indices of kpts01 in image ( 0<=x<w, 0<=y<h )
    # ids01_valid_depth: the valid indices of kpts01 with valid depth ( depth > 0 )
    z01_interpolate, kpts01_valid, ids01, ids01_valid_corners, ids01_valid_depth = interpolate_depth(uv01, depth1)

    outimage_mask = torch.ones(ids0.shape[0], device=ids0.device).bool()
    outimage_mask[ids01_valid_corners] = 0
    ids01_invalid_corners = torch.arange(0, ids0.shape[0], device=ids0.device)[outimage_mask]
    ids_outside = ids0[ids01_invalid_corners]

    # ids_valid: matched kpts01 without occlusion
    ids_valid = ids0[ids01]
    kpts0_valid = kpts0_valid[ids01]
    z01_proj = z01[ids01]

    inlier_mask = torch.abs(z01_proj - z01_interpolate) < 0.05

    # indices of kpts01 with occlusion
    ids_occlude = ids_valid[~inlier_mask]

    ids_valid = ids_valid[inlier_mask]
    if ids_valid.size(0) == 0:
        # raise ValueError('empty tensor: ids')
        raise EmptyTensorError

    kpts01_valid = kpts01_valid[inlier_mask]
    kpts0_valid = kpts0_valid[inlier_mask]

    # indices of kpts01 which are no matches in image1 for sure,
    # other projected kpts01 are not sure because of no depth in image0 or imgae1
    ids_out = torch.cat([ids_outside, ids_occlude])

    # kpts0_valid: valid keypoints0, the invalid and inconsistance keypoints are removed
    # kpts01_valid: the warped valid keypoints0
    # ids: the valid indices
    return kpts0_valid, kpts01_valid, ids_valid, ids_out



def warp(kpts0, params: dict):
    mode = params['warp_type']
    
    return warp_homography(kpts0, params)
    # elif mode == 'se3':
    #     return warp_se3(kpts0, params)
    # else:
    #     raise ValueError('unknown mode!')



def display_image_in_actual_size(image):
    import matplotlib.pyplot as plt

    dpi = 100
    height, width = image.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    if len(image.shape) == 3:
        ax.imshow(image, cmap='gray')
    elif len(image.shape) == 2:
        if image.dtype == np.uint8:
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
            ax.text(20, 20, f"Range: {image.min():g}~{image.max():g}", color='red')
    plt.show()



class EmptyTensorError(Exception):
    pass
