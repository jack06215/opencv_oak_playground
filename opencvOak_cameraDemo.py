import cv2 as cv
import depthai as dai
import numpy as np

# optional flags
extend_disparity = False
subpixel = False
lr_check = False

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - two momo cams
camLeft = pipeline.createMonoCamera()
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

camRight = pipeline.createMonoCamera()
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(640, 400)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Create a node that will produce the depth map
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)

# Set up median filter kernel
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
depth.setMedianFilter(median)

# Left-Right check
depth.setLeftRightCheck(lr_check)

# Normal disparity values range from 0..95 wil be used for normalisation
max_disparity = 95

if extend_disparity:
    max_disparity = max_disparity * 2   # double the range
if subpixel:
    max_disparity = max_disparity * 32  # 5 fractional bits (2^5)
depth.setExtendedDisparity(extend_disparity)
depth.setSubpixel(subpixel)

# When we get disparity to the host, we will multiply all values with the multiplier for better visualisation
multiplier = 255 / max_disparity
    
camLeft.out.link(depth.left)
camRight.out.link(depth.right)

# Create output
xout = pipeline.createXLinkOut()
xout.setStreamName("disparity")
depth.disparity.link(xout.input)

xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

xoutLeft = pipeline.createXLinkOut()
xoutLeft.setStreamName("cam_left")
camLeft.out.link(xoutLeft.input)

xoutRight = pipeline.createXLinkOut()
xoutRight.setStreamName("cam_right")
camRight.out.link(xoutRight.input)

with dai.Device(pipeline) as device:
    qDisparity = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qLrft = device.getOutputQueue(name="cam_left", maxSize=4, blocking=False)
    qRight = device. getOutputQueue(name="cam_right", maxSize=4, blocking=False)
    while True:
        inDepth = qDisparity.get()
        inRgb = qRgb.get()
        inLeft = qLrft.tryGet()
        inRight = qRight.tryGet()
        
        frameDepth = inDepth.getFrame()
        frameDepth = (frameDepth * multiplier).astype(np.uint8)
        frameDepth = cv.applyColorMap(frameDepth, cv.COLORMAP_JET)
        
        frameRgb = inRgb.getCvFrame()
        
        cv.imshow("RGB", frameRgb)
        cv.imshow("disparity", frameDepth)

        if inLeft is not None:
            frameLeft = inLeft.getCvFrame()
        if inRight is not None:
            frameRight = inRight.getCvFrame()
        
        if frameLeft is not None:
            cv.imshow("left", frameLeft)
        if frameRight is not None:
            cv.imshow("right", frameRight)
        
        if cv.waitKey(1) == ord('q'):
            device.close()
            break

cv.destroyAllWindows()
del pipeline