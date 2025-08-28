import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("potato.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define region to cover the entire frame
# Examples for reference:
# region_points = [(20, 400), (1080, 400)]                                      # line counting
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]              # rectangle region
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region
region_points = [(451, 618), (1120, 620), (1119, 727), (1450, 736), (1421, 145), (430, 131)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="best.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    conf=0.3,  # confidence threshold
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows