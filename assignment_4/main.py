#The sift approach was used to create the align function
#The sift approach was used to create the align function
#The sift approach was used to create the align function

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def edge_detect(reference_image):
    this_img = cv.imread(reference_image)
    img_gray = cv.cvtColor(this_img, cv.COLOR_BGR2GRAY)
    gray = np.float32(img_gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)

    this_img[dst > 0.001 * dst.max()] = [0, 0, 255]
    cv.imwrite('reference_edge_detected.jpg', this_img)
    cv.imshow('dst', this_img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

edge_detect("reference_img.png")

def align_sift(image_to_align_path, reference_image_path,
                     max_features=2000, good_match_percent=0.7,
                     min_matches=10, ransac_thresh=5.0):

    ref_color = cv.imread(image_to_align_path, cv.IMREAD_COLOR)
    img_color = cv.imread(reference_image_path, cv.IMREAD_COLOR)
    if img_color is None or ref_color is None:
        raise FileNotFoundError("Could not load one or both images.")

    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    ref_gray = cv.cvtColor(ref_color, cv.COLOR_BGR2GRAY)

    max_dim = 1000
    scale = min(max_dim / max(img_gray.shape), max_dim / max(ref_gray.shape), 1.0)
    if scale != 1.0:
        img_gray = cv.resize(img_gray, (int(img_gray.shape[1]*scale), int(img_gray.shape[0]*scale)),
                             interpolation=cv.INTER_AREA)
        ref_gray = cv.resize(ref_gray, (int(ref_gray.shape[1]*scale), int(ref_gray.shape[0]*scale)),
                             interpolation=cv.INTER_AREA)
        img_color = cv.resize(img_color, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv.INTER_AREA)
        ref_color = cv.resize(ref_color, (ref_gray.shape[1], ref_gray.shape[0]), interpolation=cv.INTER_AREA)

    sift = cv.SIFT_create(max_features)
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
    kp_img, des_img = sift.detectAndCompute(img_gray, None)

    if des_ref is None or des_img is None:
        raise ValueError("No descriptors found in one of the images.")

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=200)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    def knn_and_filter(desA, desB, ratio=good_match_percent):
        raw = flann.knnMatch(desA, desB, k=2)
        good = []
        for pair in raw:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
        return good

    good = knn_and_filter(des_ref, des_img, ratio=good_match_percent)

    aligned = None
    M = None

    def compute_homography_from_matches(good_matches, kp_query, kp_train):
        if len(good_matches) < min_matches:
            return None, None
        src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_train[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransac_thresh)
        return M, mask

    M, mask = compute_homography_from_matches(good, kp_ref, kp_img)
    if M is not None and mask is not None:
        inliers = int(mask.sum())
        inlier_ratio = inliers / len(good) if len(good) > 0 else 0.0

    if M is None or (inlier_ratio < 0.1):
        good2 = knn_and_filter(des_img, des_ref, ratio=good_match_percent)
        M2, mask2 = compute_homography_from_matches(good2, kp_img, kp_ref)
        if M2 is not None:
            try:
                Minv = np.linalg.inv(M2)
                inliers2 = int(mask2.sum())
                inlier_ratio2 = inliers2 / len(good2) if len(good2) > 0 else 0.0
                if (M is None) or (inlier_ratio2 > inlier_ratio):
                    M = Minv
                    mask = mask2
            except np.linalg.LinAlgError:
                pass

    if M is not None:
        h_img, w_img = img_gray.shape
        aligned = cv.warpPerspective(ref_color, M, (w_img, h_img),
                                     flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    else:
        aligned = cv.resize(ref_color, (img_color.shape[1], img_color.shape[0]), interpolation=cv.INTER_LINEAR)

    cv.imwrite("sift_aligned_img.jpg", aligned)

    ref_rgb = cv.cvtColor(ref_color, cv.COLOR_BGR2RGB)
    img_rgb = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)
    aligned_rgb = cv.cvtColor(aligned, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(ref_rgb); plt.title("Image to Align"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(img_rgb); plt.title("Reference Image"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(aligned_rgb); plt.title("Aligned Reference"); plt.axis('off')
    plt.tight_layout()
    plt.show()

    return aligned, ref_color, img_color

align_sift("align_this.jpg", "reference_img.png")