import cv2

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(x, y)

img = cv2.imread("pic2.png")
cv2.imshow("image", img)
cv2.setMouseCallback("image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Region points:", points)