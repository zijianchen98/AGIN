import cv2
from AmbrosioTortorelliMinimizer import *


def get_structure(img):
    result = []
    for c in cv2.split(img):
        solver = AmbrosioTortorelliMinimizer(c, alpha=1000, beta=0.01,
                                             epsilon=0.01)

        f, v = solver.minimize()
        result.append(f)

    img = cv2.merge(result)
    return img


def LFM(img):
    result = []
    for channel in cv2.split(img):
        solver = AmbrosioTortorelliMinimizer(channel, iterations=1, tol=0.1,
                                             solver_maxiterations=6)
        f, v = solver.minimize()
        result.append(f)
        # edges.append(v)

    f = cv2.merge(result)
    # v = np.maximum(*edges)
    img2 = f * 1
    cv2.normalize(img2, img2, 0, 255, cv2.NORM_MINMAX)
    img3 = np.uint8(img2)

    return img3


def show_image(image):
    img = image * 1
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = np.uint8(img)
    return img



if __name__ == '__main__':
    img = cv2.imread('../test_image/2.png', 1)

    # LFM
    result, edges = [], []
    for channel in cv2.split(img):
        solver = AmbrosioTortorelliMinimizer(channel, iterations=1, tol=0.1, solver_maxiterations=6)
        f, v = solver.minimize()
        result.append(f)
        edges.append(v)

    f = cv2.merge(result)
    LFM_result = show_image(f)
    cv2.imwrite("../output/2_lfm.png", LFM_result, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
