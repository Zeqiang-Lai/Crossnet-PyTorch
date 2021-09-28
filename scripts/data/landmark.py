import cv2
import numpy as np
import os
import pickle

def sift_knn_bbs(img1, img2):
    t1 = cv2.imread(img1,0)
    t2 = cv2.imread(img2,0)

    sift=cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(t1, None)
    kp2, des2 = sift.detectAndCompute(t2, None)

    f=cv2.drawKeypoints(t1,kp1,None,[0,0,255],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    nf=cv2.drawKeypoints(t2,kp2,None,[255,0,0],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good1 = []
    for m,n in matches:
        if m.distance < 0.3*n.distance:
            good1.append([m])

    matches = bf.knnMatch(des2,des1, k=2)

    good2 = []
    for m,n in matches:
        if m.distance < 0.3*n.distance:
            good2.append([m])

    good=[]

    for i in good1:
        img1_id1=i[0].queryIdx
        img2_id1=i[0].trainIdx

        (x1,y1)=kp1[img1_id1].pt
        (x2,y2)=kp2[img2_id1].pt

        for j in good2:
            img1_id2=j[0].queryIdx
            img2_id2=j[0].trainIdx

            (a1,b1)=kp2[img1_id2].pt
            (a2,b2)=kp1[img2_id2].pt

            if (a1 == x2 and b1 == y2) and (a2 == x1 and b2 == y1):
                good.append(i)

    match_points = []
    for g in good:
        img1_id1=g[0].queryIdx
        img2_id1=g[0].trainIdx
        (x1,y1)=kp1[img1_id1].pt
        (x2,y2)=kp2[img2_id1].pt
        match_points.append((y1,x1,y2,x2))
        
    return match_points

if __name__ == '__main__':
    img1_dir = '/media/exthdd/datasets/refsr/LF_Flowers_Dataset/refsr_rgb/img1_HR'
    img2_dir = '/media/exthdd/datasets/refsr/LF_Flowers_Dataset/refsr_rgb/img2_HR'
    target_dir = '/media/exthdd/datasets/refsr/LF_Flowers_Dataset/refsr_rgb/img1_img2_matches'
    
    names = os.listdir(img1_dir)
    os.makedirs(target_dir, exist_ok=True)
    for idx, name in enumerate(names):
        print("{}|{} {}".format(idx, len(names), name))
        img1 = os.path.join(img1_dir, name)
        img2 = os.path.join(img2_dir, name)
        match_points = sift_knn_bbs(img1, img2)
        with open(os.path.join(target_dir, name[:-4]+'.pkl'), "wb") as fp:
            pickle.dump(match_points, fp)
        