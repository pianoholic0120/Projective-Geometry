import numpy as np
import cv2
from utils import solve_homography, warping


if __name__ == '__main__':

    # ================== Part 3 ========================
    secret1 = cv2.imread('../resource/BL_secret1.png')
    secret2 = cv2.imread('../resource/BL_secret2.png')
    if secret1 is None or secret2 is None:
        print('Error: Could not load secret images.')
        exit()
    corners1 = np.array([[429, 337], [517, 314], [570, 361], [488, 380]])
    corners2 = np.array([[346, 196], [437, 161], [483, 198], [397, 229]])
    h, w, c = (500, 500, 3)
    
    ideal_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H1 = solve_homography(corners1, ideal_corners)
    dst1 = np.zeros((h, w, c), dtype=secret1.dtype)
    output3_1 = warping(secret1, dst1, H1, 0, h, 0, w, direction='b')
    H2 = solve_homography(corners2, ideal_corners)
    dst2 = np.zeros((h, w, c), dtype=secret2.dtype)
    output3_2 = warping(secret2, dst2, H2, 0, h, 0, w, direction='b')

    cv2.imwrite('output3_1.png', output3_1)
    cv2.imwrite('output3_2.png', output3_2)