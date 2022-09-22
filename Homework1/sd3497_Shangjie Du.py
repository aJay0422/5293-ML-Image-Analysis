import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN


TARGET_COLOR = np.array([131, 152, 197])


def UNI_Name_kmeans(imgPath,imgFilename,savedImgPath=None,savedImgFilename=None,k=3):
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

    # for i, mask in enumerate(masks):
    #     mask = mask.reshape(size[0], size[1], 1)
    #     cluster = img * mask
    #     print(dists[i])
    #     cv2.imshow(f"Cluster {i+1}", cluster)


    # Find the cluster which has minimum distance and plot that cluster
    cluster_idx = np.argmin(dists)
    mask = masks[cluster_idx].reshape(size[0], size[1], 1)

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

    cv2.waitKey(0)


if __name__ == "__main__":
    imgPath = "./"
    imgFilename = "faces.jpg"
    UNI_Name_kmeans(imgPath, imgFilename, k=5)
