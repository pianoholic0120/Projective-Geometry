import numpy as np
import cv2
from cv2 import aruco
from tqdm import tqdm
from utils import solve_homography, warping

def planarAR(REF_IMAGE_PATH, VIDEO_PATH):
    """
    Reuse the previously written function "solve_homography" and "warping" to implement this task
    :param REF_IMAGE_PATH: path/to/reference/image
    :param VIDEO_PATH: path/to/input/seq0.avi
    """
    video = cv2.VideoCapture(VIDEO_PATH)
    ref_image = cv2.imread(REF_IMAGE_PATH)
    if ref_image is None:
        print("Error: Could not load the reference image")
        return
    h_ref, w_ref, c_ref = ref_image.shape
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videowriter = cv2.VideoWriter("output2.avi", fourcc, film_fps, (film_w, film_h))
    arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()
    ref_corns = np.array([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32)

    pbar = tqdm(total = 353)
    while (video.isOpened()):
        ret, frame = video.read()
        if not ret:
            break
        corners, ids, _ = aruco.detectMarkers(frame, arucoDict, parameters=arucoParameters)
        if ids is not None and len(ids) > 0:
            marker_corners = corners[0].reshape(-1, 2).astype(np.float32)
            H = solve_homography(ref_corns, marker_corners)
            if H is not None:
                xmin = int(np.min(marker_corners[:, 0]))
                xmax = int(np.max(marker_corners[:, 0])) + 1
                ymin = int(np.min(marker_corners[:, 1]))
                ymax = int(np.max(marker_corners[:, 1])) + 1
                frame = warping(ref_image, frame, H, ymin, ymax, xmin, xmax, direction='b') # backward warping
            videowriter.write(frame)
            pbar.update(1)

    pbar.close()
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # ================== Part 2: Marker-based planar AR========================
    VIDEO_PATH = '../resource/seq0.mp4'
    REF_IMAGE_PATH = '../resource/hehe.jpg' 
    planarAR(REF_IMAGE_PATH, VIDEO_PATH)