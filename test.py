import sys
import numpy as np
import cv2

def main(filename):
    """ Main Method """
    img = cv2.imread(filename)

    # Obtain only red channel
    r = cv2.split(img)[2]

    # Obtain luminance from the YCrCb Colour Space
    luma_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = cv2.split(luma_image)[0]

    # Get difference
    diff_image = r
    cv2.subtract(r, y, diff_image)

    # Add threshold
    blur = cv2.GaussianBlur(diff_image, (5, 5), 0)
    thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # Remove noise through opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # For-sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding For-sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)[1]

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling; ret returns the number of apples found+1
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # Print number of apples and output image
    print(str(ret-1) + " apples found")
    cv2.imshow('image', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("ERROR: Please enter filepath")
