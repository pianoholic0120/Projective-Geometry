import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3) # initially an identity matrix
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        # feature detection & matching 
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        if des1 is None or des2 is None:
            print("No descriptors found in one of the images.")
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        num_matches = min(100, len(matches))
        matches = matches[:num_matches]
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
        # apply RANSAC to choose best H
        num_iterations = 1000
        inlier_threshold = 5.0
        best_inliers = 0
        best_H = None
        for _ in range(num_iterations):
            indices = random.sample(range(len(pts1)), 4)
            pts1_sample = pts1[indices]
            pts2_sample = pts2[indices]
            H_candidate = solve_homography(pts2_sample, pts1_sample)
            if H_candidate is None:
                continue
            pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1)))).T # (3, N)
            projected = H_candidate @ pts2_h
            projected /= projected[2, :]
            projected_pts = projected[:2, :].T # (N, 2)
            distances = np.linalg.norm(projected_pts - pts1, axis=1)
            inliers = distances < inlier_threshold
            num_inliers = np.sum(inliers)
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_H = H_candidate
        refined_H = best_H
        if refined_H is None:
            print("RANSAC failed for image pair index:", idx)
            continue

        # chain the homographies
        chain_H = last_best_H @ refined_H
        last_best_H = chain_H

        # apply warping
        h2, w2 = im2.shape[:2]
        corners_im2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        corners_im2_h = np.hstack((corners_im2, ones)).T # (3, 4)
        projected_corners = chain_H @ corners_im2_h
        projected_corners /= projected_corners[2, :]
        projected_corners = projected_corners[:2, :].T # (4, 2)

        min_x = max(int(np.floor(np.min(projected_corners[:, 0]))), 0)
        min_y = max(int(np.floor(np.min(projected_corners[:, 1]))), 0)
        max_x = min(int(np.ceil(np.max(projected_corners[:, 0]))), dst.shape[1])
        max_y = min(int(np.ceil(np.max(projected_corners[:, 1]))), dst.shape[0])

        dst = warping(im2, dst, chain_H, min_y, max_y, min_x, max_x, direction = 'b')
    
    out = dst
    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)