import cv2 as cv
import depthai as dai

# Start Pipeline
pipeline = dai.Pipeline()

# Define a source - two mono (grayscale) cameras
camLeft = pipeline.createMonoCamera()
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

camRight = pipeline.createMonoCamera()
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

xoutLeft = pipeline.createXLinkOut()
xoutLeft.setStreamName("cam_left")
camLeft.out.link(xoutLeft.input)

xoutRight = pipeline.createXLinkOut()
xoutRight.setStreamName("cam_right")
camRight.out.link(xoutRight.input)

with dai.Device(pipeline) as device:

    qLrft = device.getOutputQueue(name="cam_left", maxSize=4, blocking=False)
    qRight = device. getOutputQueue(name="cam_right", maxSize=4, blocking=False)

    frameLeft = None 
    frameRight = None
    while True:
        inLeft = qLrft.tryGet()
        inRight = qRight.tryGet()

        if inLeft is not None:
            frameLeft = inLeft.getCvFrame()
        if inRight is not None:
            frameRight = inRight.getCvFrame()
        
        if frameLeft is not None:
            cv.imshow("left", frameLeft)
        if frameRight is not None:
            cv.imshow("right", frameRight)

        if cv.waitKey(1) == ord('q'):
            break
