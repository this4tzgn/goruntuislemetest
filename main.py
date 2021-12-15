import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt


"""
#DERS1:
imager = cv2.imread("/root/image-process/datas/klon.jpg")#resmi okuma işlemi
cv2.namedWindow("show the image",cv2.WINDOW_NORMAL)#resmin penceresini resizeable yapma
[h,w] = imager.shape[:2] # en boy pixel sayısını verir
print("h is : ", h)
print("w is : ", w)
cv2.imshow("show the image",imager)#resmi gösterme ve titile verme
cv2.waitKey(0)#resmi sürekli açık kalması için
cv2.imwrite("klon2.jpg",img=imager)#resmi klon2 ismi ile farklı kaydet
cv2.destroyAllWindows()

"""



"""
#DERS3:
def resizeTHEwindow(img,height = None,width= None,inter = cv2.INTER_AREA):
    (h,w) = img.shape[:2]
    if height == None and width == None:
        return img
    if width == None:
        r = height/float(h)
        dim = (int(w*r),height)
    else:
        r = width/float(w)
        dim = (width,int(h*r))
    return cv2.resize(img,dim,interpolation=inter)
def learnDIM(img):
    [h, w] = img.shape[:2]  # en boy pixel sayısını verir
    print("h is : ", h)
    print("w is : ", w)

imager = cv2.imread("/root/image-process/datas/klon.jpg")#resmi tekrar oku
imager = resizeTHEwindow(imager,height=600,width=600) # yeniden boyutlandırma gerçekleşti
learnDIM(imager)#resim boyutlarını öğrenme
cv2.imshow("clone",imager)#yeni resmi göster
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

"""
#DERS 4: video açma işlemleri

cap = cv2.VideoCapture(0)

while True:
    enable,frame = cap.read() # frame geliyorsa enable 1 olur ve gelen frame'ler frame değişkenine kaydolur
    if enable == 0:
        break
    frame = cv2.flip(frame,1)

    cv2.imshow("webcam",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

"""

"""
#DERS 5: video kaydetme işlemleri
cap = cv2.VideoCapture(0)#video kaydetmeye başla

filepath = "/root/image-process/webcam_rec_0.avi" #kayıt adresi
codec = cv2.VideoWriter_fourcc("W","M","V","2") # .avi uzantısı için codec
frameRate = 30 #fps
resolution = (640,480) #çözünürlük 
output = cv2.VideoWriter(filepath,codec,frameRate,resolution) #ayarları dosyaya uygulama

while True:
    enable , frame = cap.read() #video verilerini çek
    frame = cv2.flip(frame,1) #frame'i y eksenine göre mirror et
    output.write(frame) #ayar dosyasına frame'i yaz
    cv2.imshow("LIVE",frame) #frame'leri bana göster
    if cv2.waitKey(1) & 0xFF == ord("q"): #q ile kapat
        break
output.release() #out dosyasını sonlandır
cap.release() #kamerayı sonlandır
cv2.destroyAllWindows() #tüm pencereleri kapat

"""

"""
#DERS 5: boş çizim tuvali açma
canvas = np.zeros((512,512,3),dtype=np.uint8)+100 #512,512 pixel bir RGB(0,0,0) arkaplan oluştur ve +100 ile RGB(100,100,100) olarak düzenle
cv2.imshow("CANVAS",canvas) #tuvali bana göster
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
#DERS 6: tuvale çarpraz çizgiler çizerek boyama işlemi 
img = np.zeros((100,100,3),dtype=np.uint8) ##canvas oluşturulması,10x10 3kanal(R,G,B), tek kanal için default olarak da 1 girilir (siyah beyaz)
for i in range(0,100):
    img[i,i] = (255,255,255) # sağa çarpraz boyama
    img[i,99-i] = (255,255 ,255) #sağa çarpraz boyama

img = cv2.resize(img,(900,900),interpolation=cv2.INTER_AREA) # pixel boyutunu artırma
cv2.imshow("canvas",img)#pencere açma
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
#DERS 7 Şekil çizme
canvas = np.zeros((512,512,3),dtype=np.uint8)
cv2.line(canvas,(0,0),(512,512),(255,255,255),thickness=5) # 0,0 noktasından başla 512,512 noktasına dek git beyaz çizgi çiz kalınlığı 5 olsun
cv2.line(canvas,(512,0),(0,512),(255,255,255),thickness=5)
cv2.rectangle(canvas,(50,50),(462,462),(0,0,255),thickness=5) # dikdörtgenin üst sol köşe kordinati 50,50 ; alt sağ köşe koordinatı 462,462
                                                # !!!!dikdörtgenin içinin dolu olması için thicknes = -1 !!!!!!
cv2.circle(canvas,(256,256),200,(255,0,0),thickness=5) #çember çizdirme
#üçgen çizdirme
p1 = (80,80)
p2= (150,70)
p3 = (80,140)
cv2.line(canvas,p1,p2,(0,255,0),thickness=5)
cv2.line(canvas,p2,p3,(0,255,0),thickness=5)
cv2.line(canvas,p3,p1,(0,255,0),thickness=5)

#poligon çizdirme
points = np.array([[[10,150],[100,100],[280,290],[300,250]]])
cv2.polylines(img=canvas,pts=points,isClosed=True,color=(0,255,0),thickness=5)
cv2.imshow("canvas",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
"""
#DERS 8 yazı yazma
 
canvas = np.zeros((512,512,3),dtype=np.uint8) +255 #beyaz tuval

font1 = cv2.FONT_ITALIC
font2 = cv2.FONT_HERSHEY_SIMPLEX
font3 = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img=canvas,text="test text",org=(30,90),fontFace=font3,fontScale=3,color=(0,255,0),thickness=1,lineType=cv2.LINE_AA)

cv2.imshow(winname="canvas",mat=canvas)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()

"""

"""
# trackbar oluşturma saçmalığı????
def null_func(): #trackbar oluşturabilmek için kullanmak gerekiyor
    pass

img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow("image")

cv2.createTrackbar("R","image",0,255,null_func) # image penceresinde 0-255 arası kızak oluştur
cv2.createTrackbar("G","image",0,255,null_func)
cv2.createTrackbar("B","image",0,255,null_func)

switch = "0:KAPALI,1:ACIK"
cv2.createTrackbar(switch,"image",0,1,null_func)
while True:
    cv2.imshow("image",img)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break
    red = cv2.getTrackbarPos("R","iamge")
    green = cv2.getTrackbarPos("G","image")
    blue = cv2.getTrackbarPos("B","image")
    s = cv2.getTrackbarPos(switch,"image")
    if s == 0:
        img[:] = [0,0,0]
    if s == 1:
        img[:] = [blue,green,red]

cv2.destroyAllWindows()

"""

"""
## pixellere erişme ve değişiklik yapma
mat = cv2.imread("datas/klon.jpg")


#piksel değeri bulma:
color = mat[12,12]
print("colors: ",color)

blue = mat[12,12,0]
print("blue: ",blue)

green = mat[12,12,1]
print("green: ",green)

red = mat[12,12,2]
print("red: ",red)

#resmin pixel adedi
dimension = mat.shape
print("resmin pixelleri: {h}x{w} {k} kanal".format(h=dimension[0],w=dimension[1],k=dimension[2]))


#pixel değiştirme
for i in range(12,72):
    for j in range(12,72):
        mat.itemset((i,j,0),0)
        mat.itemset((i,j,1),255)
        mat.itemset((i,j,2),0)


cv2.imshow("klon",mat)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""


"""
#roi oluşturma
#roi >>>> region of interest >>>> ilgilenilen alan
mat = cv2.imread("datas/klon.jpg")
roi = mat[30:200,200:400] # y ekseninde 30'dan 200. pixele dek tara; x ekseninde 200'den 400'e dek tara
cv2.imshow("klon",mat)
cv2.imshow("ROI",roi )
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
"""
#resimleri toplama işlemi

circle = np.zeros((512,512,3),dtype=np.uint8) +128 # tuval rengi gri olsun
cv2.circle(img=circle,center=(256,256),radius=60,color=(255,0,0),thickness=-1) # -1 dairenin içini doldurmak için

rect = np.zeros((512,512,3),dtype=np.uint8) +128
cv2.rectangle(img=rect,pt1=(150,150),pt2=(350,350),color=(0,255,0),thickness=-1)

#resimleri toplama
add = cv2.add(src1=circle,src2=rect)

#ağırlıklı toplama:
dst = cv2.addWeighted(circle,0.3,rect,0.7,0) #yüzde 30 circle yüzde 70 kareden al

cv2.imshow(winname="circle",mat=circle)
cv2.imshow(winname="rectangle",mat=rect)
cv2.imshow(winname="plusser",mat=add)
cv2.imshow(winname="weighed",mat=dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
"""
#resimde renk uzayı değiştirme:
mat = cv2.imread(filename="datas/klon.jpg")

mat2rgb = cv2.cvtColor(src=mat,code=cv2.COLOR_BGR2RGB)
mat2hsv = cv2.cvtColor(src=mat,code=cv2.COLOR_BGR2HSV)
mat2gray = cv2.cvtColor(src=mat,code=cv2.COLOR_BGR2GRAY)


cv2.imshow(winname="BGR",mat=mat)
cv2.imshow(winname="HSV",mat=mat2hsv)
cv2.imshow(winname="RGB",mat=mat2rgb)
cv2.imshow(winname="GRAY",mat=mat2gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
#videoda renk uzayı dönüştürme
cap = cv2.VideoCapture(0)

while True:
    enable,frame = cap.read() # frame geliyorsa enable 1 olur ve gelen frame'ler frame değişkenine kaydolur
    if enable == 0:
        break
    frame = cv2.flip(frame,1)
    frame_bgr2hsv = cv2.cvtColor(src=frame,code=cv2.COLOR_BGR2HSV)

    cv2.imshow("NORMAL",frame)
    cv2.imshow(winname="TO HSV",mat=frame_bgr2hsv)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

"""
"""
#Renk filtreleme için hsv kodlarını elde etme

cap =  cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow(winname="Trackbar")
cv2.resizeWindow(winname="Trackbar",width=500,height=500)

cv2.createTrackbar("Lower - H","Trackbar",0,180,nothing)
cv2.createTrackbar("Lower - S","Trackbar",0,255,nothing)
cv2.createTrackbar("Lower - V","Trackbar",0,255,nothing)

cv2.createTrackbar("Upper - H","Trackbar",0,180,nothing)
cv2.createTrackbar("Upper - S","Trackbar",0,255,nothing)
cv2.createTrackbar("Upper - V","Trackbar",0,255,nothing)

################### upper kızaklarını max seviyede başlat
cv2.setTrackbarPos(trackbarname="Upper - H",winname="Trackbar",pos=180)
cv2.setTrackbarPos(trackbarname="Upper - S",winname="Trackbar",pos=255)
cv2.setTrackbarPos(trackbarname="Upper - V",winname="Trackbar",pos=255)

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
#################HSV formatına çevir
    frame_hsv = cv2.cvtColor(src=frame,code=cv2.COLOR_BGR2HSV)

###############Kızak değerlerini al
    lower_h = cv2.getTrackbarPos(trackbarname="Lower - H",winname="Trackbar")
    lower_s = cv2.getTrackbarPos(trackbarname="Lower - S", winname="Trackbar")
    lower_v = cv2.getTrackbarPos(trackbarname="Lower - V", winname="Trackbar")

    upper_h = cv2.getTrackbarPos(trackbarname="Upper - H", winname="Trackbar")
    upper_s = cv2.getTrackbarPos(trackbarname="Upper - S", winname="Trackbar")
    upper_v = cv2.getTrackbarPos(trackbarname="Upper - V", winname="Trackbar")

    lower_color = np.array([lower_h,lower_s,lower_v])
    upper_color = np.array([upper_h,upper_s,upper_v])

    mask = cv2.inRange(src=frame_hsv,lowerb=lower_color,upperb=upper_color)

    cv2.imshow(winname="NORMAL",mat=frame)
    cv2.imshow(winname="MASK",mat=mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

"""

"""
#filtreler

img_filter = cv2.imread("datas/filter.png")
img_median = cv2.imread("datas/median.png")
img_bilateral = cv2.imread("datas/bilateral.png")

blur = cv2.blur(src=img_filter,ksize=(7,7))
blur_g = cv2.GaussianBlur(src=img_filter,ksize=(5,5),sigmaX=cv2.BORDER_DEFAULT)
blur_m = cv2.medianBlur(src=img_median,ksize=5)
blur_b = cv2.bilateralFilter(src=img_bilateral,d=9,sigmaColor=95,sigmaSpace=95)

cv2.imshow(winname="blur",mat=blur)
cv2.imshow(winname="blur_g",mat=blur_g)
cv2.imshow(winname="blur_m",mat=blur_m)
cv2.imshow(winname="blur_b",mat=blur_b)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""

"""
mat1 = cv2.imread(filename="datas/bitwise_1.png")
mat2 = cv2.imread(filename="datas/bitwise_2.png")

bit_and = cv2.bitwise_and(src1=mat1,src2=mat2)
bit_or = cv2.bitwise_or(src1=mat1,src2=mat2)
bit_not1 = cv2.bitwise_not(src=mat1)
bit_not2 = cv2.bitwise_not(src=mat2)
bit_xor = cv2.bitwise_xor(src1=mat1,src2=mat2)

cv2.imshow(winname="bit pic 1",mat=mat1)
cv2.imshow(winname="bit pic 2",mat=mat2)
cv2.imshow(winname="bit and",mat=bit_and)
cv2.imshow(winname="bit or",mat=bit_or)
cv2.imshow(winname="bit xor",mat=bit_xor)
cv2.imshow(winname="bit not 1",mat=bit_not1)
cv2.imshow(winname="bit not 2",mat=bit_not2)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

"""
#erosion,gradient vs
cap = cv2.VideoCapture(0)
kernel = np.ones((2,2),np.uint8)
while True:
    enable,frame = cap.read() # frame geliyorsa enable 1 olur ve gelen frame'ler frame değişkenine kaydolur
    if enable == 0:
        break
    frame = cv2.flip(frame,1)
    frametogray = cv2.cvtColor(src=frame,code=cv2.COLOR_BGR2GRAY)
    frame_modified = cv2.morphologyEx(src=frametogray,op=cv2.MORPH_GRADIENT,kernel=kernel)

    cv2.imshow("NORMAL",frame)
    cv2.imshow(winname="gradient",mat=frame_modified)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

"""

cap = cv2.VideoCapture(0)
while True:
    enable,frame = cap.read() # frame geliyorsa enable 1 olur ve gelen frame'ler frame değişkenine kaydolur
    if enable == 0:
        break
    frame = cv2.flip(frame,1)

    cv2.imshow("NORMAL",frame)
    plt.close(1)
    plt.figure(1)
    plt.hist(x=frame.ravel(),bins=256,range=[0,256])
    plt.show()

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()


