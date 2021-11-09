import  numpy as np
import  cv2
import random

def Find_Match(des1,des2):
    FLANN_INDEX_KDTREE = 1
    param1 = dict(algorithm=FLANN_INDEX_KDTREE, trees=20)
    param2 = dict(checks=100)
    flann = cv2.FlannBasedMatcher(param1,param2)
    matches = flann.knnMatch(des1,des2,k=2)
    Goodpoints = []
    for i,j in matches:
        if i.distance < 0.75*j.distance:
            Goodpoints.append(i)    
            Pts1 = np.int32([ KeyPoint1[i.queryIdx].pt for i in Goodpoints ]).reshape(-1,2)
            Pts2 = np.int32([ KeyPoint2[i.trainIdx].pt for i in Goodpoints ]).reshape(-1,2)    
    return Pts1,Pts2

def Find_Epipole(f):
    B = -f[:,-1]  
    A = f[:,:-1]     
    status,e = cv2.solve(A,B,flags=cv2.DECOMP_SVD)    
    return e 

def Draw_line(points,lines,img):
    row,col,dim = img.shape
    points = np.uint16(points)
    for i in range(np.size(points,axis=1)):
        img = cv2.circle(img,tuple([points[0,i],points[1,i]]),10,(0,255,0),15)
        x1 = 0
        y1 = np.uint16(-lines[2,i]/lines[1,i])
        x2 = col-1
        y2 = np.uint16(-(lines[2,i]+lines[0,i]*x2)/lines[1,i])
        img = cv2.line(img,tuple([x1,y1]),tuple([x2,y2]),(255,0,0),3)
    return img

Img1 = cv2.imread('Img1.JPG')
Img2 = cv2.imread('Img2.JPG') 
row,col,dim = Img1.shape
sift = cv2.SIFT_create()
KeyPoint1,Descriptor1 = sift.detectAndCompute(Img1,None)
KeyPoint2,Descriptor2 = sift.detectAndCompute(Img2,None)
Pts1,Pts2 =  Find_Match(Descriptor1,Descriptor2)   
FundamentalMatrix, mask = cv2.findFundamentalMat(Pts1,Pts2,cv2.FM_LMEDS)  
Inlier1 = Img1.copy()
Inlier2 = Img2.copy()
for i in range(np.size(Pts1, axis=0)):
    if mask[i]==1:
        Inlier1 = cv2.circle(Inlier1,tuple(Pts1[i]),5,(0,255,0),5)
        Inlier2 = cv2.circle(Inlier2,tuple(Pts2[i]),5,(0,255,0),5)
    if mask[i]==0:
        Inlier1 = cv2.circle(Inlier1,tuple(Pts1[i]),5,(0,0,255) ,5)
        Inlier2 = cv2.circle(Inlier2,tuple(Pts2[i]),5,(0,0,255) ,5)  
Inlier = np.concatenate((Inlier1,Inlier2),axis=1)
cv2.imwrite('Inlier-and-Outlier.jpg',Inlier)
Epipole1 = np.int32(Find_Epipole(FundamentalMatrix))
Epipole2 = np.int32(Find_Epipole(FundamentalMatrix.T))
ratio = 10
Bigrow = ((row+abs(Epipole1[1]))[0]+1000)//ratio
Bigcol = ((col+abs(Epipole1[0]))[0]+1000)//ratio
EpipoleImage1 = np.ones([Bigrow,Bigcol,3])*255
Shiftrow = (500-Epipole1[1][0])//ratio
Shiftcol = (500-Epipole1[0][0])//ratio
EpipoleImage1 = cv2.circle(EpipoleImage1,tuple([Epipole1[0][0]//ratio+Shiftcol,Epipole1[1][0]//ratio+Shiftrow]),5,(0,0,255),5)
Small_Im = cv2.resize(Img1, (col//ratio,row//ratio),interpolation = cv2.INTER_NEAREST)
EpipoleImage1[Shiftrow:Shiftrow+row//ratio,Shiftcol:Shiftcol+col//ratio] = Small_Im
cv2.imwrite('Epipole_Img1.jpg',EpipoleImage1)
Bigrow = ((row+abs(Epipole2[1]))[0]+1000)//ratio
Bigcol = ((abs(Epipole2[0]))[0]+1000)//ratio
EpipoleImage2 = np.ones([Bigrow,Bigcol,3])*255
Shiftrow = (500-Epipole2[1][0])//ratio
Shiftcol = (500)//ratio
EpipoleImage2 = cv2.circle(EpipoleImage2,tuple([Epipole2[0][0]//ratio+Shiftcol,Epipole2[1][0]//ratio+Shiftrow]),2,(0,0,255),5)
Small_Im = cv2.resize(Img2, (col//ratio,row//ratio),interpolation = cv2.INTER_NEAREST)
EpipoleImage2[Shiftrow:Shiftrow+row//ratio,Shiftcol:Shiftcol+col//ratio] = Small_Im
cv2.imwrite('Epipole_Img2.jpg',EpipoleImage2)
InlierPoints1 = np.ones([3,10])
InlierPoints2 = np.ones([3,10])
counter = 0
while True:
    indx = random.randint(1,np.size(mask)-1)
    if mask[indx,0]==1:
        InlierPoints1[:2,counter] = Pts1[indx]
        InlierPoints2[:2,counter] = Pts2[indx]
        counter = counter+1
        if counter==10:
            break        
EpipolarLines1 = np.matmul(FundamentalMatrix,InlierPoints1)
EpipolarIm1 = Draw_line(InlierPoints2,EpipolarLines1,Img2)
EpipolarLines2 = np.matmul(FundamentalMatrix.T,InlierPoints2)
EpipolarIm2 = Draw_line(InlierPoints1,EpipolarLines2,Img1)
EpipolarIm = np.concatenate((EpipolarIm2,EpipolarIm1),axis=1)
cv2.imwrite('Final_img.jpg',EpipolarIm)