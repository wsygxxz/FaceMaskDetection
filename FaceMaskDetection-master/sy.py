
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# 读取模板图像
# image = cv2.imread("D:\py\FaceMaskDetection-master\img/reference.png")
image = cv2.imread("D:\py\FaceMaskDetection-master\img/reference2.png")
image_gary = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)  # 转换成灰度图
print(image.shape)

# 初始化一个与原图像等同的矩阵
temp = np.zeros((417,640))
# temp = np.zeros((255,386))
temp = temp.astype(np.uint8)

# 查找图像中的矩阵
ret,thresh = cv2.threshold(image_gary, 250, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(thresh, 2, 1)
img0,contours, hierarchy = cv2.findContours(thresh, 2, 1)
# print(contours, hierarchy)
cnt=contours[1]
x, y, w, h = cv2.boundingRect(cnt)
print(x,y,w,h)
img = cv2.rectangle(image, (x-30, y-10), (x + 100*w, y - 90*h), (0, 255, 0), 2)
cv2.imshow("contour2.jpg", img)
# 显示水印图片
# image2 = cv2.imread("D:\py\FaceMaskDetection-master\img/watermark.png")
image2 = cv2.imread("D:\py\FaceMaskDetection-master\img/watermark2.png")
roi = image2[y:y+h,x:x+w,0:3]
# roi = image2[y-10: y - 90*h,x-30:x + 100*w,0:3]
roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)
ret,roi = cv2.threshold(roi, 60, 100, cv2.THRESH_BINARY)
roi = cv2.morphologyEx(roi,cv2.MORPH_ELLIPSE,(5,5))
cv2.imshow("roi",roi)


# 将水印图片赋值给初始化的矩阵图片
roi2 = temp[y:y+h,x:x+w]
roi3 = cv2.add(roi, roi2)
temp[y:y+h,x:x+w] = roi3
cv2.imshow("temp",temp)

dst = cv2.inpaint(image2, temp, 30, cv2.INPAINT_NS) # 使用INPAINT_TELEA算法进行修复
cv2.imshow('TELEA', dst)
cv2.waitKey(0)
