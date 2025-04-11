import numpy as np


def solve_homography(u, v):
    """
    Compute the Homography matrix H, such that v ≈ H * u.
    This function uses the Direct Linear Transformation (DLT) method, and fixes h33 = 1.
    
    Parameters:
        u: numpy array of shape (N x 2), each row represents a point [u_x, u_y] in the source image
        v: numpy array of shape (N x 2), each row represents the corresponding point [v_x, v_y] in the target image
    Returns:
        H: 3x3 Homography matrix, such that v ≈ H * u (using homogeneous coordinates)
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    u_x = u[:, 0]
    u_y = u[:, 1]
    v_x = v[:, 0]
    v_y = v[:, 1]
    ones_N = np.ones(N)
    
    # (1) h11*u_x + h12*u_y + h13 - v_x*(h31*u_x + h32*u_y + 1) = 0
    # (2) h21*u_x + h22*u_y + h23 - v_y*(h31*u_x + h32*u_y + 1) = 0
    # Arrange the unknowns as x = [h11,h12,h13, h21,h22,h23, h31,h32], and fix h33 = 1.
    # The equations can be written as A*x = b, where A is a (2N x 8) matrix and b is a (2N,) vector.
    A = np.zeros((2 * N, 8))
    b_vec = np.zeros(2 * N)
    
    # Odd rows (corresponding to v_x equation)
    A[0::2, 0] = u_x
    A[0::2, 1] = u_y
    A[0::2, 2] = ones_N
    A[0::2, 6] = -v_x * u_x
    A[0::2, 7] = -v_x * u_y
    b_vec[0::2] = v_x
    
    # Even rows (corresponding to v_y equation)
    A[1::2, 3] = u_x
    A[1::2, 4] = u_y
    A[1::2, 5] = ones_N
    A[1::2, 6] = -v_y * u_x
    A[1::2, 7] = -v_y * u_y
    b_vec[1::2] = v_y

    # Solve the least squares problem Ax = b
    x, residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
    # x contains: [h11, h12, h13, h21, h22, h23, h31, h32]

    # solve H with A
    H = np.array([[x[0], x[1], x[2]],
                  [x[3], x[4], x[5]],
                  [x[6], x[7], 1.0]])

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    1. Backward warping (direction='b'):
    2. Forward warping (direction='f'):

    Parameters:
        src: Source image, with shape (h_src, w_src, channels)
        dst: Destination image, with shape (h_dst, w_dst, channels)
        H: 3x3 Homography matrix
        ymin, ymax, xmin, xmax: Define the warp region (if direction='b', it represents the target image region; if direction='f', it represents the source image region)
        direction: 'b' for backward warp (default), 'f' for forward warp
    
    Returns:
        The warped destination image, dst
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)
    xs = np.arange(xmin, xmax)
    ys = np.arange(ymin, ymax)
    X, Y = np.meshgrid(xs, ys, sparse = False)  # X, Y shape = (region_h, region_w)
    ones_arr = np.ones(X.size)
    if direction == 'b':
        coords_dst = np.vstack((X.ravel(), Y.ravel(), ones_arr))  # (3, N)
        coords_src = H_inv @ coords_dst
        # Apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        coords_src /= coords_src[2, :]  # Homogeneous coordinate normalization
        X_src = coords_src[0, :]
        Y_src = coords_src[1, :]
        # Filter out the positions that correspond to valid locations in the source image (within [0, w_src-1] and [0, h_src-1] due to rounding requiring a 4-neighborhood)
        valid_mask = (X_src >= 0) & (X_src < w_src - 1) & (Y_src >= 0) & (Y_src < h_src - 1)
        region_h, region_w = Y.shape
        warped_region = np.zeros((region_h * region_w, ch), dtype=src.dtype)
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size > 0:
            X_src_valid = X_src[valid_mask]
            Y_src_valid = Y_src[valid_mask]
            x0 = np.round(X_src_valid).astype(np.int32)
            y0 = np.round(Y_src_valid).astype(np.int32)
            
            I = src[y0, x0, :]  # shape (N_valid, ch)
            
            warped_region[valid_indices] = I
        region_h, region_w = Y.shape
        warped_img = warped_region.reshape((region_h, region_w, ch))
        dst_region = dst[ymin:ymax, xmin:xmax]
        valid_mask_reshaped = valid_mask.reshape((region_h, region_w))
        dst_region[valid_mask_reshaped] = warped_img[valid_mask_reshaped]
        dst[ymin:ymax, xmin:xmax] = dst_region

    elif direction == 'f':
        # Forward warp: Map points in the source image to the destination position (using nearest neighbor interpolation)
        coords_src = np.vstack((X.ravel(), Y.ravel(), ones_arr))
        coords_dst = H @ coords_src
        coords_dst /= coords_dst[2, :]
        X_dst = coords_dst[0, :]
        Y_dst = coords_dst[1, :]
        X_dst_round = np.round(X_dst).astype(np.int32)
        Y_dst_round = np.round(Y_dst).astype(np.int32)       
        valid_mask = (X_dst_round >= 0) & (X_dst_round < w_dst) & (Y_dst_round >= 0) & (Y_dst_round < h_dst)
        if np.any(valid_mask):
            X_src_valid = X.ravel()[valid_mask]
            Y_src_valid = Y.ravel()[valid_mask]
            X_dst_valid = X_dst_round[valid_mask]
            Y_dst_valid = Y_dst_round[valid_mask]
            dst[Y_dst_valid, X_dst_valid] = src[Y_src_valid, X_src_valid]
    
    return dst 

