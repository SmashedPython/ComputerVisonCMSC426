import numpy as np
import cv2
from UndistortImage import UndistortImage
import os
import matplotlib.pyplot as plt

def ReadCameraModel(models_dir):

# ReadCameraModel - load camera intrisics and undistortion LUT from disk
# INPUTS:
#   image_dir: directory containing images for which camera model is required
#   models_dir: directory containing camera models
#
# OUTPUTS:
#   fx: horizontal focal length in pixels
#   fy: vertical focal length in pixels
#   cx: horizontal principal point in pixels
#   cy: vertical principal point in pixels
#   G_camera_image: transform that maps from image coordinates to the base
#     frame of the camera. For monocular cameras, this is simply a rotation.
#     For stereo camera, this is a rotation and a translation to the left-most
#     lense.
#   LUT: undistortion lookup table. For an image of size w x h, LUT will be an
#     array of size [w x h, 2], with a (u,v) pair for each pixel. Maps pixels
#     in the undistorted image to pixels in the distorted image
    intrinsics_path = models_dir + "/stereo_narrow_left.txt"
    lut_path = models_dir + "/stereo_narrow_left_distortion_lut.bin"

    intrinsics = np.loadtxt(intrinsics_path)
    # Intrinsics
    fx = intrinsics[0,0]
    fy = intrinsics[0,1]
    cx = intrinsics[0,2]
    cy = intrinsics[0,3]
    # 4x4 matrix that transforms x-forward coordinate frame at camera origin and image frame for specific lens
    G_camera_image = intrinsics[1:5,0:4]
    # LUT for undistortion
    # LUT consists of (u,v) pair for each pixel)
    lut = np.fromfile(lut_path, np.double)
    lut = lut.reshape([2, lut.size//2])
    LUT = lut.transpose()

    return fx, fy, cx, cy, G_camera_image, LUT


def compute_intrinsic_matrix(fx, fy, cx, cy):
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K

def load_demosaic_images(filename,LUT):
    img = cv2.imread(filename,flags=-1)
    color_image = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    undistorted_image = UndistortImage(color_image,LUT)
    return undistorted_image

# SIFT Algorithm: How to Use SIFT for Image Matching in Python
def sift_feature_track(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray,None)
    return keypoints, descriptors

def feature_matching(descriptors1,descriptors2):
    # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    # matches = bf.match(descriptors1,descriptors2)
    # matches = sorted(matches, key = lambda x:x.distance)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) 

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors1,descriptors2,k=2)

    good_matches = [m for m, n in matches if m.distance < 0.78 * n.distance]

    return good_matches

def compute_fundamental_matrix(keypoints1, keypoints2):
    fundamental_matrix, mask = cv2.findFundamentalMat(keypoints1, keypoints2, cv2.FM_RANSAC)
    return fundamental_matrix

def compute_essential_matrix(F, K):
    Kt = np.transpose(K)
    E = np.dot(np.dot(Kt, F), K)
    E = E / np.linalg.norm(E)
    return E

def decompose_essential_matrix(E, points1, points2, K):
    _, R, t, _ = cv2.recoverPose(E, points1, points2, K)
    return R, t

def reconstruct_trajectory(rotations, translations):
    trajectory = [np.zeros((3, 1))]
    
    Rt_acc = np.identity(4)

    origin = np.vstack((np.zeros((3,1)), np.array([1])))
    for R, t in zip(rotations, translations):
        t = t.reshape(3, 1)
        Rt = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
        Rt_acc = Rt_acc @ Rt
        new_pos = Rt_acc @ origin

        trajectory.append(new_pos[:3, 0].reshape(3,1))
    return np.array(trajectory)

def plot_trajectory(trajectory):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='o', linewidth=5)
    ax.set_title('3D Reconstruction of Camera Trajectory')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()

def process():
    models_dir = 'Oxford_dataset_reduced/model'
    fx, fy, cx, cy, _, LUT = ReadCameraModel(models_dir= models_dir)
    K = compute_intrinsic_matrix(fx, fy, cx, cy)

    images_dir = 'Oxford_dataset_reduced/images'
    processed_images = []

    file_list = os.listdir(images_dir)
    file_list.sort()

    count = 0 # For small sample check
    for file in file_list:
        file_path = os.path.join(images_dir, file)
        processed_images.append(load_demosaic_images(file_path, LUT))

        # count += 1
        # if count == 3:
        #   break


    keypoints_list = []
    descriptors_list = []
    for p in processed_images:
        keypoints, descriptors = sift_feature_track(p)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    R_list = []
    T_list = []
    for i in range(len(descriptors_list) - 1):
        matches = feature_matching(descriptors1 = descriptors_list[i],descriptors2=descriptors_list[i+1])
        points1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints_list[i+1][m.trainIdx].pt for m in matches])

        ''' Regular compute fundamental matrix function'''
        F = compute_fundamental_matrix(keypoints1=points1,keypoints2=points2)
        ''' My compute fundamental matrix function'''
        # F = my_computeF(keypoints1=points1,keypoints2=points2,)

        E = compute_essential_matrix(F,K)

        ''' Regular decompose essential matrix function'''
        # R, t = decompose_essential_matrix(E, points1, points2, K)
        ''' My  decompose essential matrix function'''
        R,t = my_decompose_E(E, points1, points2, K)

        R_list.append(R)
        T_list.append(t)

    ''' Store the Rotations and Translation data '''
    np.save('rotations.npy', R_list)  
    np.save('translations.npy', T_list) 

    trajectory = reconstruct_trajectory(R_list, T_list)
    plot_trajectory(trajectory)


'''
Bounus part Hand calculate F
'''
def normalize(x, y):

  x = x.astype(np.float64)
  y = y.astype(np.float64)
    
  mean_x = np.mean(x)
  mean_y = np.mean(y)
  x -= mean_x
  y -= mean_y
  scale = np.sqrt(2) / np.mean(np.sqrt(x**2 + y**2))
  T = np.array([[scale, 0, -scale * mean_x],
                [0, scale, -scale * mean_y],
                [0, 0, 1]])
  x_norm = x * scale
  y_norm = y * scale
  norm_p = np.column_stack((x_norm, y_norm))

  return norm_p, T

def handcomputeF(x1, y1, x2, y2):
  """
  Function: compute fundamental matrix from corresponding points
  Input:
     x1, y1, x2, y2 - coordinates
  Output:
     fundamental matrix, 3x3
  """
  # 1. Make matrix A
  n = len(x1)
  A = np.zeros((n, 9))
  # homogeneous equation
  for i in range(n):
    A[i] = [x1[i]*x2[i], x1[i]*y2[i], x1[i], y1[i]*x2[i], y1[i]*y2[i], y1[i], x2[i], y2[i], 1]
  # 2. Do SVD for A
  U, S, Vt = np.linalg.svd(A)
  # 3. Find fundamental matrix F
  F = Vt[-1].reshape(3, 3).T
  
  U, S, Vt = np.linalg.svd(F)
  S[-1] = 0
  F = U @ np.diag(S) @ Vt
    
  return F

def getInliers(x1, y1, x2, y2, F, thresh):
  """
   Function: implement the criteria checking inliers.
   Input:
     x1, y1, x2, y2 - coordinates
     F - estimated fundamental matrix, 3x3
     thresh - threshold for passing the error
   Output:
     inlier indices
  """
  inliers = []
  for i in range(len(x1)):
    p1 = np.array([x1[i], y1[i], 1])
    p2 = np.array([x2[i], y2[i], 1])  
    # 1. Compute epipolar lines. Here a line is expressed as a vector
    line = F @ p1   
    line2 = F.T @ p2
    # 2. Calculate the distances of points in the second image from the corresponding lines
    dist = np.abs(p2 @ line) / np.sqrt(line[0]**2 + line[1]**2)
    # 3. Check distances with the threshold and find inliers
    if dist < thresh:
      inliers.append(i)
  return inliers

def ransacF(x1, y1, x2, y2, num_iterations=10, threshold=0.01):

  max_inliers = []
  best_F = None

  for _ in range(num_iterations):
  #    1. Randomly select 8 points
    indices = np.random.choice(len(x1), 8, replace=False)
    pts1 = np.column_stack((x1[indices], y1[indices]))
    pts2 = np.column_stack((x2[indices], y2[indices]))
  #    2. Call computeF()
    F = handcomputeF(pts1[:, 0], pts1[:, 1], pts2[:, 0], pts2[:, 1])
  #    3. Call getInliers()
    inliers = getInliers(x1, y1, x2, y2, F, threshold)
  #    4. Update F and inliers.
    if len(inliers) > len(max_inliers):
      max_inliers = inliers
      best_F = F
  return best_F, inliers

def my_computeF(keypoints1, keypoints2):
    # Ensure that keypoints are numpy arrays
    p1 = np.array([kp.pt for kp in keypoints1],dtype = np.float32)
    p2 = np.array([kp.pt for kp in keypoints1],dtype = np.float32)

    px1, py1 = p1[:, :1], p1[:, 1:2]
    px2, py2 = p2[:, :1], p2[:, 1:2]

    # Normalize points
    norm_p1, T1 = normalize(px1, py1)
    norm_x1, norm_y1 = norm_p1[:, :1], norm_p1[:, 1:2]

    norm_p2, T2 = normalize(px2, py2)
    norm_x2, norm_y2 = norm_p2[:, :1], norm_p2[:, 1:2]

    # Compute fundamental matrix with computeF()
    F_ours, Inliers = ransacF(norm_x1, norm_y1, norm_x2, norm_y2)
    F_ours = np.matmul(np.matmul(np.transpose(T2),F_ours), T1)
    F_ours = F_ours/F_ours[2,2]
    return F_ours


''' 
    Bonus points for Hand decompose essential matrix
'''

def decompose_essential_matrix(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    return (R1, R2), t

def create_camera_matrices(R1, R2, t):
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]

def triangulate_points(P1, P2, pts1, pts2):
    pts1_homogeneous = cv2.convertPointsToHomogeneous(pts1)[:, 0, :-1]
    pts2_homogeneous = cv2.convertPointsToHomogeneous(pts2)[:, 0, :-1]

    points_4d = cv2.triangulatePoints(P1, P2, pts1_homogeneous.T, pts2_homogeneous.T)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)[:, 0, :]
    return points_3d

def find_correct_camera_pose(Rs, ts, K, pts1, pts2):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera matrix for the first camera
    
    max_valid_points = 0
    best_pose = None

    for R, t in create_camera_matrices(Rs[0], Rs[1], ts):
        P2 = K @ np.hstack((R, t.reshape(3, 1)))  # Camera matrix for the second camera
        points_3d = triangulate_points(P1, P2, pts1, pts2)
        points_3d_cam2 = (R @ points_3d.T).T + t

        valid_points = np.sum((points_3d[:, 2] > 0) & (points_3d_cam2[:, 2] > 0))

        if valid_points > max_valid_points:
            max_valid_points = valid_points
            best_pose = (R, t)

    return best_pose

def my_decompose_E(E, pts1, pts2, K):
    Rs, ts = decompose_essential_matrix(E)
    return find_correct_camera_pose(Rs, ts, K, pts1, pts2)

''' 
    start
'''
process()

rotations = np.load('rotations.npy', allow_pickle=True)
translations = np.load('translations.npy', allow_pickle=True)

trajectory = reconstruct_trajectory(rotations, translations)
plot_trajectory(trajectory)

