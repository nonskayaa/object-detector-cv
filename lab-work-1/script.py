import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image_4.jpg') 

blurred = cv2.GaussianBlur(image, (5, 5), 0)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened_kernel = cv2.filter2D(image, -1, kernel)
blurred_for_sharpening = cv2.GaussianBlur(image, (5, 5), 0)
sharpened_unsharp = cv2.addWeighted(image, 1.5, blurred_for_sharpening, -0.5, 0)
sharpened = sharpened_unsharp

edges = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
edges = cv2.convertScaleAbs(edges)

combined = cv2.addWeighted(blurred, 0.5, edges, 0.5, 0)

combined = cv2.addWeighted(combined, 0.5, sharpened, 0.5, 0)

def show_images(original, blurred, edges, sharpened, combined):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 3, 1)
    plt.title('Оригинал')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)

    plt.title('Размытие по Гауссу')
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Выделение границ')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Повышение резкости ')

    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Комбинация изображений ')
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

show_images(image, blurred, edges, sharpened, combined)
