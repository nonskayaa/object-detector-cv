import cv2
import numpy as np

image_path = "image_4.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

object_data = []
for contour in contours:
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = cv2.contourArea(contour)
        object_data.append({'center': (cx, cy), 'area': area, 'contour': contour})
    else:
        object_data.append({'center': (0, 0), 'area': 0, 'contour': contour})

object_data = sorted(object_data, key=lambda x: x['area'], reverse=True)

if len(object_data) > 0:
    largest_object = object_data[0]
    smallest_object = object_data[-1]
    largest_center = largest_object['center']
    smallest_center = smallest_object['center']

    cv2.putText(image, f"largest: {largest_center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, f"smallest: {smallest_center}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
else:
    print("No objects detected!")

for obj in object_data:
    cv2.drawContours(image, [obj['contour']], -1, (255, 0, 0), 2)
    cx, cy = obj['center']
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

cv2.imshow("Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
