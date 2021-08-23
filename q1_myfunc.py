import cv2
import numpy as np
import os

from numpy.lib.function_base import copy


def show(im4,height=800):
    (h4,w4)=np.shape(im4)[0:2]
    scale=height/h4
    dim=(int(scale * w4) , height)
    im4_resize=cv2.resize(im4.copy() , dim )
    cv2.imshow('tmp', im4_resize)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    







   
def calc_pan_size(myH,frames):
    size=(9500,5000)
    corners=[]
    [h1 , w1 , _] = np.shape(frames[450-1])
    pnt1 = np.array([[0,0], [0, h1],[w1, h1], [w1, 0]],np.float64).reshape(-1, 1, 2)
    tmp=np.array([(0,0,1),(0,h1,1),(w1,h1,1),(w1,0,1)])

    for i in [0,899]:
        im_second=frames[i].copy()
        H=myH[i]
        pnt2=[]
        for row in tmp:
            p=np.matmul(H,np.reshape(row,(3,1)))
            p=(p/p[2])
            p=((p[0][0]),(p[1][0]))
            pnt2.append(p)
        pnt2=np.array(pnt2,np.float64)
        pnt2=np.array(np.reshape(pnt2,(-1,1,2)),np.float64)
        pnts=np.concatenate((pnt1,pnt2))

        [i_min , j_min]=np.floor(pnts.min(axis=0).tolist()[0])
        [i_max , j_max]=np.ceil(pnts.max(axis=0).tolist()[0])

        trans_dist = [-i_min,-j_min]
        th=int(trans_dist[1])
        tw=int(trans_dist[0])

        translate=np.array([[1 , 0 , -tw+size[0]/2-w1/1.1],
                            [0 , 1 , -th+size[1]/2-1.3*h1],
                            [0 , 0 , 1]])

        H_trans=np.array([[1,0,tw],
                          [0,1,th],
                          [0,0,1]])
        H2=np.matmul(H_trans,H)
        pan2=cv2.warpAffine(cv2.warpPerspective(im_second,H2,size),translate[0:2],size)

        corners_tmp=[]
        for row in pnt1:
            p=np.matmul(H2,np.reshape(np.reshape(np.concatenate((row,[1]),axis=None),(3,1)),(3,1)))
            p=(np.matmul(translate,p))
            p=(p/p[2])  
            corners_tmp.append((int(p[0][0]),int(p[1][0])))
        corners_tmp=np.array(corners_tmp,np.float64)
        corners.append(np.array(np.reshape(corners_tmp,(-1,1,2)),np.float64))


    c1=np.max(np.max(corners,axis=1),axis=0)
    c2=np.min(np.min(corners,axis=1),axis=0)
    size=(np.uint64((c1-c2)-(50,50)).tolist()[0])
    size[1]=size[1]-size[1]%8+8
    size[0]=size[0]-size[0]%12+12
    size=tuple(size)
    return size


import copy




def myhomography(im1,im2):

    MIN_MATCH_COUNT = 10
    img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)



    if len(good)>MIN_MATCH_COUNT:
        src_pt = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pt = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


    M, mask = cv2.findHomography(src_pt, dst_pt, cv2.RANSAC,5.0)
    return M



#########H=myhomography(im_second,im_base)


def findpan(img_base,img_second,right=False):
    im_base=img_base.copy()
    im_second=img_second.copy()

    H=myhomography(im_second,im_base)

    [h1 , w1 , _] = np.shape(im_base)
    [h2 , w2 , _] = np.shape(im_second)

    pnt1 = np.array([[0,0], [0, h1],[w1, h1], [w1, 0]],np.float32).reshape(-1, 1, 2)

    tmp=np.array([(0,0,1),(0,h2,1),(w2,h2,1),(w2,0,1)])
    pnt2=[]
    for row in tmp:
        p=np.matmul(H,np.reshape(row,(3,1)))
        p=(p/p[2])
        p=((p[0][0]),(p[1][0]))
        pnt2.append(p)

    pnt2=np.array(pnt2,np.float32)
    pnt2=np.array(np.reshape(pnt2,(-1,1,2)),np.float32)

    pnts=np.concatenate((pnt1,pnt2))


    [i_min , j_min]=np.floor(pnts.min(axis=0).tolist()[0])
    [i_max , j_max]=np.ceil(pnts.max(axis=0).tolist()[0])

    trans_dist = [-i_min,-j_min]
    th=int(trans_dist[1])
    tw=int(trans_dist[0])
    H_trans=np.array([[1 , 0 , tw],
                        [0 , 1 , th],
                        [0 , 0 , 1]])
    H2=np.matmul(H_trans,H)

    size=(int(i_max+np.abs(i_min)) ,  int(j_max+np.abs(j_min)))

    pan=cv2.warpPerspective(im_second,H2,size)

    mask=np.ones([size[1],size[0],3])
    mask_margin=22
    
    mask[th+mask_margin:th+h1-mask_margin, tw+mask_margin:tw+w1-mask_margin]=0
    gausskernel=cv2.getGaussianKernel(ksize=(2*mask_margin+1),sigma=15)
    mask=cv2.sepFilter2D(mask,ddepth=cv2.CV_64F,kernelX=gausskernel,kernelY=gausskernel)
    if not right:
        mask[th:th+h1, tw+w1-2*mask_margin:]=0
    
    pan_tmp=np.zeros([size[1],size[0],3])
    pan_tmp[th:th+h1, tw:tw+w1] = im_base

    pan=np.multiply(pan,mask)
    pan_tmp=np.multiply(pan_tmp,1-mask)
    pan=np.uint8(np.add(pan_tmp,pan))


    return pan



import pickle
def saveVar(myvar,name,direc=True):
    if direc:
        path = os.path.join(os.getcwd(), 'variables')
        try:
            os.mkdir(path)
        except:
            pass
        
        name='variables/'+name+'.pckl'  
    else:
        name=name+'.pckl' 
    f = open(name, 'wb')
    pickle.dump(myvar, f)
    f.close()
    return
    

    
def readvar(name):
    name='variables/'+name+'.pckl'  
    f = open(name, 'rb')
    myvar = pickle.load(f)
    f.close()
    return myvar
    

    
    
        
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    