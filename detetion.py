import cv2
import numpy as np
import pytesseract


def getNumber(path, num=3):
    '''
    获取需要识别的文字
    :param path: 图片路径
    :param num: 缩放倍数
    :return: 图像中的蓝色字体
    '''
    #读入图片
    invoice = cv2.imread(path)
    #图像缩放
    size = invoice.shape
    width = int(size[0]/num)
    lenth = int(size[1]/num)
    invoice = cv2.resize(invoice, (lenth, width), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(invoice, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 48, 48])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing


def tiaobian(tiaobian,  num=50):
    '''
    跳变检测
    :param tiaobian: 二值图像
    :return: 跳变检测结构
    '''
    size = tiaobian.shape
    width = size[0]
    lenth = size[1]
    for i in range(width):
        for j in range(lenth):
            if tiaobian[i, j] == 255 and j + num < lenth:
                for k in range(j, j + num):
                    if tiaobian[i, k] == 255:
                        for m in range(j, k):
                            tiaobian[i, m] = 255
    return tiaobian


def Detect(img):
    '''
    文字检测
    :param img: 二值图像
    :return: 文字轮廓位置
    '''
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lst = []
    for k in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[k])
        lst.append([x-5,y-5,x+w+5,y+h])
    return lst


def imgCrop(img, y1, y2, x1, x2):
    crop_img = img[x1:x2, y1:y2]
    #img[纵向范围：横向范围]
    cv2.imwrite('./image/'+str(y1)+str(x1)+'.png', crop_img)
    #以左上角坐标为名存入image文件夹下
    return crop_img

invoice = cv2.imread('4.jpg')
size = invoice.shape
width = int(size[0]/3)
lenth = int(size[1]/3)
invoice = cv2.resize(invoice, (lenth, width), interpolation=cv2.INTER_CUBIC)
location = Detect(tiaobian(getNumber('4.jpg')))
for k in location:
    #左上角坐标(k[0], k[1])，右下角坐标(k[2], k[3])
    print(pytesseract.image_to_string(imgCrop(invoice, k[0], k[2], k[1], k[3]), lang='chi_sim'))
    # img = cv2.rectangle(invoice, (k[0], k[1]), (k[2], k[3]), (0, 255, 0), 1)

# cv2.imshow('invoice', img)
# cv2.waitKey()
# cv2.destroyAllWindows()


# cv2.imshow('tiaobian', getNumber('fapiao.jpg'))
# cv2.waitKey()
# cv2.destroyAllWindows()
