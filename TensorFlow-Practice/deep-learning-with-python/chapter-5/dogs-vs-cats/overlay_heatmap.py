import numpy as np
import cv2

heatmap = cv2.imread('heatmap.jpeg')
elephants = cv2.imread('two_elephants.jpeg')

heatmap = cv2.resize(heatmap, (elephants.shape[1], elephants.shape[0]))
heatmap = np.uint8(255*heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

alpha = 0.6
beta = (1.0 - alpha)
superimposed = cv2.addWeighted(elephants, alpha, heatmap, beta, 0.0)

cv2.imshow('elephants', elephants)
cv2.imshow('heatmap', heatmap)
cv2.imshow('superimposed', superimposed)
cv2.waitKey()

# we can get some more clarity by concentrating the heatmap
heatmap = np.square(heatmap)
heatmap = heatmap/ heatmap.max()
heatmap = np.uint8(255*heatmap)
superimposed = cv2.addWeighted(elephants, alpha, heatmap, beta, 0.0)
cv2.imshow('superimposed', superimposed)
cv2.waitKey()


