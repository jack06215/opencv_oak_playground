import cv2 as cv
import depthai as dai

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Create output
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# Connect to the device
with dai.Device() as device:
    # Print out available cameras
    print('Connected cameras: ', device.getConnectedCameras())
    # Start pipeline
    device.startPipeline(pipeline)

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

        # Retrieve 'bgr' (opencv format) frame
        frame = inRgb.getCvFrame()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("bgr", frame)
        cv.imshow("grayscale", frame_gray)

        if cv.waitKey(1) == ord('q'):
            break