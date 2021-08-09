import cv2 as cv
import numpy as np


def callback(x):
    global H_low, H_high, S_low, S_high, V_low, V_high

    H_low = cv.getTrackbarPos('low H', 'controls')
    H_high = cv.getTrackbarPos('high H', 'controls')
    S_low = cv.getTrackbarPos('low S', 'controls')
    S_high = cv.getTrackbarPos('high S', 'controls')
    V_low = cv.getTrackbarPos('low V', 'controls')
    V_high = cv.getTrackbarPos('high V', 'controls')


def dilatation(src):
    dilatation_size = cv.getTrackbarPos('Kernel size', 'controls')
    dilation_shape = morph_shape(cv.getTrackbarPos('Element shape', 'controls'))
    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv.dilate(src, element)
    return dilatation_dst


def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE


def detect(frame, lower_limit, upper_limit):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    maskUndilated = cv.inRange(hsv, lower_limit, upper_limit)
    mask = dilatation(maskUndilated)

    res = cv.bitwise_and(frame, frame, mask=mask)
    M = cv.moments(mask)
    if M['m00'] > 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    else:
        cX, cY = 0, 0

    cv.circle(res, (cX, cY), 5, (255, 0, 0), -1)

    return res, maskUndilated, mask
