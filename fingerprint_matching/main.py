import cv2 as cv
import numpy as np


def preprocess_fingerprint(image_path):
    img = cv.imread(image_path, 0)
    _, img_bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return img_bin


def orb_bf_match(img1_path, img2_path):
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)

    orb_detector = cv.ORB_create(nfeatures=1000)

    keypoint1, descriptor1 = orb_detector.detectAndCompute(img1, None)
    keypoint2, descriptor2 = orb_detector.detectAndCompute(img2, None)

    brutef = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    # KNN Match
    matches = brutef.knnMatch(descriptor1, descriptor2, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Draw matches
    match_img = cv.drawMatches(img1, keypoint1, img2, keypoint2, good_matches, None,
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), match_img


# Test with run_orb_bf_matching

def run_orb_bf_matching(img1_path, img2_path, threshold):
    match_count, match_img = orb_bf_match(img1_path, img2_path)

    if match_count > threshold:
        print("Match found")
        cv.imwrite(f'orb_bf_match_&{img1_path}.jpg', match_img)
        return match_img
    else:
        print("No match found")
        return 0


def sift_flann_match(img1_path, img2_path):
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)

    sift_detector = cv.SIFT_create(nfeatures=1000)

    keypoint1, descriptor1 = sift_detector.detectAndCompute(img1, None)
    keypoint2, descriptor2 = sift_detector.detectAndCompute(img2, None)

    # Check if descriptors were found
    if descriptor1 is None or descriptor2 is None:
        print("No descriptors found in one or both images")
        return 0, None

    flan_index = 1
    index = dict(algorithm=flan_index, trees=5)
    search = dict(checks=50)

    flann_match = cv.FlannBasedMatcher(index, search)

    antall_matches = flann_match.knnMatch(descriptor1, descriptor2, k=2)

    # Apply Lowe's ratio test - CORRECTED VERSION
    matched = []
    for match_pair in antall_matches:
        if len(match_pair) == 2:  # Ensure we have two matches
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                matched.append(m)

    # Draw matches
    if len(matched) > 0:
        match_img = cv.drawMatches(img1, keypoint1, img2, keypoint2, matched, None,
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        # Create a blank image if no matches found
        height = max(img1.shape[0], img2.shape[0])
        width = img1.shape[1] + img2.shape[1]
        match_img = np.zeros((height, width), dtype=np.uint8)

    return len(matched), match_img


def run_sift_flann_matching(img1_path, img2_path, threshold):
    match_count, match_img = sift_flann_match(img1_path, img2_path)

    if match_count > threshold:
        print("Match found")
        cv.imwrite(f'sift_flann_match_&{img1_path}.jpg', match_img)
        return match_img
    else:
        print("No match found")
        return 0


run_orb_bf_matching("UiA1.jpg", "UiA2.png", 10)
run_sift_flann_matching("UiA1.jpg", "UiA2.png", 10)