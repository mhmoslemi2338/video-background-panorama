import timeit
start =timeit.default_timer()


import cv2
import numpy as np
from q1_myfunc import show , findpan , saveVar , readvar , myhomography , calc_pan_size


######## import video frames ######

myvideo = cv2.VideoCapture('video.mp4')
frames=[]
count=0
while True :      
    if count==900:
        break
    _ , tmp_frame=myvideo.read()
    frames.append(tmp_frame)
    count+=1
        
del myvideo
del tmp_frame
saveVar(frames, 'frames')


##### select frame 270 and 450 ######
frame90=frames[90-1]
frame270=frames[270-1]
frame450=frames[450-1]
frame630=frames[630-1]
frame810=frames[810-1]

pan2=findpan(img_second=frame90,img_base=frame270,right=False)
pan3=findpan(img_second=frame810,img_base=frame630,right=True)

pan4=findpan(img_second=pan2,img_base=frame450,right=False)
panf=findpan(img_second=pan3,img_base=pan4,right=True)

cv2.imwrite('key frame panorama.jpg',panf);

del pan2;  del pan3; del pan4; del panf;
del frame90; del frame270; del frame630; del frame810;

print('done!')

#%%
#*****************************************************#
#******************* PART 3 & 4 **********************#
#*****************************************************#


print('PART 3 & 4 : ')


frame100=frames[100-1].copy()
frame180=frames[180-1].copy()
frame300=frames[300-1].copy()
frame450=frames[450-1].copy()
frame600=frames[600-1].copy()
frame750=frames[750-1].copy()
frame820=frames[820-1].copy()

H100_180=myhomography(frame100,frame180)
H180_300=myhomography(frame180,frame300)
H750_600=myhomography(frame750,frame600)
H820_750=myhomography(frame820,frame750)


H300_450=myhomography(frame300,frame450)
H600_450=myhomography(frame600,frame450)
H180_450=np.matmul(H180_300,H300_450)
H100_450=np.matmul(H180_450,H100_180)
H750_450=np.matmul(H750_600,H600_450)
H820_450=np.matmul(H750_450,H820_750)


###########################################

print( "\n calc homography for all frames ( progress %) : ")
myH=[]

for i in range(900):

    frame_reduce=frames[i].copy()
    if (i+1)%90==0:
        print(int((i+1)/9),end=' ')

    
    if i<100-1:
        H=myhomography(frame_reduce,frame100)
        H=np.matmul(H,H100_450)
        myH.append(H)
        
    elif i==100-1:
        myH.append(H100_450)
        
    elif i<180-1:
        H=myhomography(frame_reduce,frame180)
        H=np.matmul(H,H180_450)
        myH.append(H)
        
    elif i==180-1:
        myH.append(H180_450)
        
    elif i<300-1:
        H=myhomography(frame_reduce,frame300)
        H=np.matmul(H,H300_450)
        myH.append(H)
        
    elif i==300-1:
        myH.append(H300_450)
        
    elif i<600-1:
        H=myhomography(frame_reduce,frame450)
        myH.append(H)
        
    elif i==600-1:
        myH.append(H600_450)
        
    elif i<750-1:
        H=myhomography(frame_reduce,frame600)
        H=np.matmul(H,H600_450)
        myH.append(H)

    elif i==750-1:
        myH.append(H750_450)
        
    elif i<820-1:
        H=myhomography(frame_reduce,frame750)
        H=np.matmul(H,H750_450)
        myH.append(H)

    elif i==820-1:
        myH.append(H820_450)
        
    else:
        H=myhomography(frame_reduce,frame820)
        H=np.matmul(H,H820_450)
        myH.append(H)
        
    
     
saveVar(myH, 'myH')


size=calc_pan_size(myH,frames)



###########################################

print("\n clac panorama for all frames (progres %): ")

(w_pan,h_pan)=size

h_pan=int(np.ceil(h_pan/8))
w_pan=int(np.ceil(w_pan/12))

fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('res05-reference-plane.mp4',fourcc, 30, size)


matrix=[]

[h1 , w1 , _] = np.shape(frames[450-1])
pnt1 = np.array([[0,0], [0, h1],[w1, h1], [w1, 0]],np.float64).reshape(-1, 1, 2)
tmp=np.array([(0,0,1),(0,h1,1),(w1,h1,1),(w1,0,1)])

mypan=[]



for i in range(900):
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
    
    translate=np.array([[1 , 0 , -tw+size[0]/2-w1/1.6],
                        [0 , 1 , -th+size[1]/2-h1/1.4],
                        [0 , 0 , 1]])
    
    H_trans=np.array([[1,0,tw],
                      [0,1,th],
                      [0,0,1]])
    H2=np.matmul(H_trans,H)

    pan2=cv2.warpPerspective(im_second,H2,size)
    pan2=cv2.warpAffine(pan2,translate[0:2],size)

    out.write(pan2)
     
    for l1 in range(8):
        for l2 in range(12):
            matrix.append(pan2[l1*h_pan:(l1+1)*h_pan,l2*w_pan:(l2+1)*w_pan])


    if (i+1)%225==0: 
        for k1 in range(96):
            tmp_var=[]
            for k2 in range(225):
                tmp_var.append(matrix[k2*96+k1])
            name='A'+str(k1+1)+'_'+str(i+1)
            saveVar(tmp_var, name)
        matrix=[]

    if (i+1)%90==0:
        print(int((i+1)/9),end=' ')

out.release()
    
del pan2
del tmp_var
del matrix
del out
del frames




###########################################
print(" \n find back-ground panorama (progress %) : ")

panorama=[]
for i in range(96):
    if int((i+1)/.96)%10 ==0:
        print(int((i+1)/.96) ,end= ' ')
    for j in [225,450,675,900]:
        name='A'+str(i+1)+'_'+str(j)
        if j==225:
            mypan=np.asarray(readvar(name))
        else:
            mypan=np.concatenate((mypan,np.asarray(readvar(name))))    
    y = np.ma.masked_where(mypan == 0, mypan)
    del mypan
    area=np.uint8(np.ma.median(y, axis=0).filled(0))
    panorama.append(area)
    







rows=[]
for j in range(8):
    s,s2=[],[]
    for i in range(6):
        s.append(cv2.hconcat([panorama[2*i+j*12],panorama[2*i+1+j*12]]))
    for i in range(3):
        s2.append(cv2.hconcat([s[2*i],s[2*i+1]]))
    rows.append(cv2.hconcat([s2[0],s2[1],s2[2]]))

result=cv2.vconcat([rows[0],rows[1]])
for i in range(6):
    result=cv2.vconcat([result,rows[2+i]])
    
    
    



cv2.imwrite('res06-background-panorama.jpg',result)


import shutil
shutil.rmtree('variables')






print( "\n make backgroung video ( progress %) : ")


[h1p,w1p]=size
center=tuple(map(int,np.multiply(0.5,np.shape(result)[0:2]).tolist()))
rot_mat = cv2.getRotationMatrix2D(center, 2 , 1.0)

pnt1 = np.array([[0,0], [0, h1p],[w1p, h1p], [w1p, 0]],np.float64).reshape(-1, 1, 2)
tmp=np.array([(0,0,1),(0,h1p,1),(w1p,h1p,1),(w1p,0,1)])

fourcc = cv2.VideoWriter_fourcc(*'avc1')
out2 = cv2.VideoWriter('res07-background-video.mp4',fourcc, 30, (w1,h1))


for i in range(900): 
    H=myH[i]
    H_inv=np.linalg.inv(H)
    pnt2=[]
    for row in tmp:
        p=np.matmul(H_inv,np.reshape(row,(3,1)))
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
    
    translate=np.array([[1 , 0 , -tw+size[0]/2-1.88*w1],
                        [0 , 1 , -th+size[1]/2-1.22*h1],
                        [0 , 0 , 1]])
    
    
    H_trans=np.array([[1,0,tw],
                      [0,1,th],
                      [0,0,1]])
    
    H_inv=np.matmul(H_trans,H_inv)
     
    tmp2=cv2.warpPerspective(result,H_inv,size)
    tmp2=cv2.warpAffine(tmp2,translate[0:2],(w1,h1))
    
    tmp2 = cv2.warpAffine(tmp2, rot_mat, tmp2.shape[1::-1], flags=cv2.INTER_LINEAR)
    

    out2.write(tmp2)
    if (i+1)%90==0:
        print(int((i+1)/9),end=' ')

out2.release()

   



print( "\n make wide backgroung video ( progress %) : ")

[h1p,w1p]=size
center=tuple(map(int,np.multiply(0.5,np.shape(result)[0:2]).tolist()))
rot_mat = cv2.getRotationMatrix2D(center, 2 , 1.0)

pnt1 = np.array([[0,0], [0, h1p],[w1p, h1p], [w1p, 0]],np.float64).reshape(-1, 1, 2)
tmp=np.array([(0,0,1),(0,h1p,1),(w1p,h1p,1),(w1p,0,1)])

fourcc = cv2.VideoWriter_fourcc(*'avc1')
out3 = cv2.VideoWriter('res09-background-video-wider.mp4',fourcc, 30, (2*w1,h1))


for i in range(0,900,2): 
    H=np.multiply(myH[i]+myH[i+1],0.5)
    H_inv=np.linalg.inv(H)
    pnt2=[]
    for row in tmp:
        p=np.matmul(H_inv,np.reshape(row,(3,1)))
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
    
    translate=np.array([[1 , 0 , -tw+size[0]/2-1.88*w1],
                        [0 , 1 , -th+size[1]/2-1.22*h1],
                        [0 , 0 , 1]])
    
    
    H_trans=np.array([[1,0,tw],
                      [0,1,th],
                      [0,0,1]])
    
    H_inv=np.matmul(H_trans,H_inv)
     
    tmp2=cv2.warpPerspective(result,H_inv,size)
    tmp2=cv2.warpAffine(tmp2,translate[0:2],(2*w1,h1))
    
    tmp2 = cv2.warpAffine(tmp2, rot_mat, tmp2.shape[1::-1], flags=cv2.INTER_LINEAR)

    out3.write(tmp2)
    
    if(int(i/2)+1)%45==0:    
        print(int((int(i/2)+1)/4.5),end=' ')
    

out3.release()

   

    

stop=timeit.default_timer()
print('\n Run-time : ' , int((stop-start)/60) , ' minutes')









