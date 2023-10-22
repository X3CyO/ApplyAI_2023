import cv2 as cv

for i in range(2):  # Check up to _# camera indices
    camera = cv.VideoCapture(i)
    if not camera.isOpened():
        continue

    print(f"Camera Index {i}:")
    print(f"Width: {camera.get(cv.CAP_PROP_FRAME_WIDTH)}")
    print(f"Height: {camera.get(cv.CAP_PROP_FRAME_HEIGHT)}")

    # Check and print supported frame rates
    for fps in range(10, 61, 10):  # adjust the range as needed
        camera.set(cv.CAP_PROP_FPS, fps)
        actual_frame_rate = camera.get(cv.CAP_PROP_FPS)
        print(f"Desired Frame Rate: {fps} FPS, Actual Frame Rate: {actual_frame_rate} FPS")

    # Release the camera
    camera.release()
