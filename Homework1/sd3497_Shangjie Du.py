import cv2
import numpy as np
import os
from sklearn.cluster import KMeans, DBSCAN


TARGET_COLOR = np.array([131, 152, 197])


def UNI_Name_kmeans(imgPath, imgFilename, savedImgPath=None, savedImgFilename=None, k=3):
    """
    K-means method for segmentation.
    :param imgPath: the path of the image folder. Please use relative path
    :param imgFilename: the name of the image file
    :param savedImgPath: the path of the folder you will save the image
    :param savedImgFilename: the name of the output image
    :param k: the number of clusters of the k-means function
    :return:
    """
	# Read the image and conver it into LAB color space
    img = cv2.imread(imgPath + imgFilename)
    size = img.shape
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Perform KMeans on AB dimension
    X = img_lab[:,:,1:].reshape(size[0] * size[1], -1)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans = kmeans.fit(X)
    cls = kmeans.predict(X)   # cluster index for each pixel

    # Compute the cluster mean for each cluster
    masks = [(cls == i) for i in range(k)]
    cluster_means = []
    for i, mask in enumerate(masks):
        cluster = img.reshape(size[0] * size[1], 3)[mask]
        cluster_means.append(cluster.mean(axis=0))

    # Compute the distance between cluster means and the target vector
    dists = []
    for cluster_mean in cluster_means:
        dist = np.linalg.norm(cluster_mean - TARGET_COLOR, ord=1)
        dists.append(dist)

    # Find the cluster which has minimum distance and plot that cluster
    cluster_idx = np.argmin(dists)
    mask = masks[cluster_idx].reshape(size[0], size[1], 1)
    cv2.imshow("The Cluster", img * mask)
    cv2.imwrite("md_images/mask.jpg", img * mask)

    # Find the bounding box
    coords = []
    for i in range(size[0]):
        for j in range(size[1]):
            if mask[i,j]:
                coords.append([i, j])
    coords = np.array(coords)
    db = DBSCAN(eps=2**0.5)
    db.fit(coords)
    face_coords = coords[db.labels_ == 0]
    box = (np.min(face_coords[:,1]),
           np.min(face_coords[:,0]),
           np.max(face_coords[:,1]),
           np.max(face_coords[:,0]))

    # Plot original image with bounding box
    img_box = img.copy()
    img_box = cv2.rectangle(img_box, [box[0], box[1]], [box[2], box[3]], color=(0, 0, 255), thickness=2)
    cv2.imshow("Bounding Box", img_box)

    # Plot segmentation
    mask_new = np.zeros((size[0], size[1]))
    for coord in face_coords:
        mask_new[coord[0], coord[1]] = 1
    mask_new = mask_new.reshape(size[0], size[1], 1)
    mask_new = mask_new.astype("uint8")
    cv2.imshow("Segmentation", img * mask_new)
    cv2.imwrite("md_images/Face Cluster.jpg", img * mask_new)

    if not os.path.exists(savedImgPath):
        os.mkdir(savedImgPath)
    cv2.imwrite(savedImgPath + savedImgFilename, img_box)

    cv2.waitKey()




def multi_faces(imgPath, imgFileName):
    import mediapipe as mp


    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(model_selection=1)

    img = cv2.imread(imgPath + imgFilename)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_detection.process(imgRGB)

    if result.detections:   # at least one face is detected
        for id, detection in enumerate(result.detections):
            ih, iw, ic = img.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, color=(255, 0, 255), thickness=2)



    cv2.imshow("faces", img)
    cv2.waitKey(0)



if __name__ == "__main__":
    imgPath = "./"
    imgFilename = "face_d2.jpg"
    savedImgPath = "./results/"
    savedImgfilename = "face_d2_result.jpg"
    k = 3
    UNI_Name_kmeans(imgPath, imgFilename, savedImgPath, savedImgfilename, k=k)