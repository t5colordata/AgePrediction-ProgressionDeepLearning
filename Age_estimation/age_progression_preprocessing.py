import os 

files=os.listdir('.')
label=[]
X_train1=[]
X_train2=[]
X_train1[:]=[]
X_train2[:]=[]

for i in files:
   X_train1.append('./'+i)
   for j in files:
      if int(i[0:3])==int(j[0:3]):
         label.append(1)
         X_train2.append('./'+j)
      else:
         label.append(0)
         X_train2.append('./'+j)


file_X_train1=open('age_progression_x_train1.txt','w')
file_X_train2=open('age_progression_x_train2.txt','w')
file_label=open('age_progression_label.txt','w')

for i in X_train1:
    file_X_train1.write(i)
    file_X_train1.write('\n')
    file_X_train1.flush()


for i in X_train2:
    file_X_train2.write(i)
    file_X_train2.write('\n')
    file_X_train2.flush()


for i in label:
    file_label.write(i)
    file_label.write('\n')
    file_label.flush()


X_train1_face=[]
X_train1_face[:]=[]
processed_x_train1_face=[]
processed_x_train1_face[:]=[]
size=120,120
for i in range(0,len(X_train1)):
   img=cv2.imread(X_train1[i])
   img=cv2.resize(img,(256,256))
   faceCascade=cv2.CascadeClassifier('1.xml')
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
   try:
      x,y,w,h=faces[0]
      crop_img=gray[y:y+h,x:x+w]
      processed_x_train1_face.append(X_train1[i])
      pil_image=Image.fromarray(crop_img)
      pl=pil_image.resize(size,Image.ANTIALIAS)
      #pil_image.thumbnail(size,Image.ANTIALIAS)
      data=np.asarray( pl, dtype="int32")
      data.shape
      X_train1_face.append(data)
   except:
      print ''


X_train2_face=[]
X_train2_face[:]=[]
processed_x_train2_face=[]
processed_x_train2_face[:]=[]
size=120,120
for i in range(0,len(X_train2)):
   img=cv2.imread(X_train2[i])
   img=cv2.resize(img,(256,256))
   faceCascade=cv2.CascadeClassifier('1.xml')
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
   try:
      x,y,w,h=faces[0]
      crop_img=gray[y:y+h,x:x+w]
      processed_x_train2_face.append(X_train1[i])
      pil_image=Image.fromarray(crop_img)
      pl=pil_image.resize(size,Image.ANTIALIAS)
      #pil_image.thumbnail(size,Image.ANTIALIAS)
      data=np.asarray( pl, dtype="int32")
      data.shape
      X_train2_face.append(data)
   except:
      print ''



X_test1_face=[]
X_test1_face[:]=[]
processed_x_test1_face=[]
processed_x_test1_face[:]=[]
size=120,120
for i in range(0,len(X_train1)):
   img=cv2.imread(X_train1[i])
   img=cv2.resize(img,(256,256))
   faceCascade=cv2.CascadeClassifier('1.xml')
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
   try:
      x,y,w,h=faces[0]
      crop_img=gray[y:y+h,x:x+w]
      processed_x_test1_face.append(X_train1[i])
      pil_image=Image.fromarray(crop_img)
      pl=pil_image.resize(size,Image.ANTIALIAS)
      #pil_image.thumbnail(size,Image.ANTIALIAS)
      data=np.asarray( pl, dtype="int32")
      data.shape
      X_test1_face.append(data)
   except:
      print ''


X_test2_face=[]
X_test2_face[:]=[]
processed_x_test2_face=[]
processed_x_test2_face[:]=[]
size=120,120
for i in range(0,len(X_train1)):
   img=cv2.imread(X_train1[i])
   img=cv2.resize(img,(256,256))
   faceCascade=cv2.CascadeClassifier('1.xml')
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
   try:
      x,y,w,h=faces[0]
      crop_img=gray[y:y+h,x:x+w]
      processed_x_test2_face.append(X_train1[i])
      pil_image=Image.fromarray(crop_img)
      pl=pil_image.resize(size,Image.ANTIALIAS)
      #pil_image.thumbnail(size,Image.ANTIALIAS)
      data=np.asarray( pl, dtype="int32")
      data.shape
      X_test1_face.append(data)
   except:
      print ''
