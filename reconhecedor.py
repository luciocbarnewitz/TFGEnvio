import cv2

imagem1 = cv2.imread('fig_amostras/fig1_amostra.png')

classificador1 = cv2.CascadeClassifier('cascade_particulas2.xml')

imagemcinza1 = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)

deteccoes1 = classificador1.detectMultiScale(imagemcinza1, scaleFactor=1.33, minNeighbors= 10)

count = 0

for (x, y, l, a) in deteccoes1:
    cv2.rectangle(imagem1, (x, y), (x + l, y + a), (0,255,0), 2)
    count += 1

print ('Quantidade de particulas: ', count)

cv2.imshow('Detector de particulas 1', imagem1)

cv2.waitKey(0)
cv2.destroyAllWindows()



