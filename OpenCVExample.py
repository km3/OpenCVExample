import cv2 as cv
import numpy as np
import argparse

# const
COLOR_WHITE  = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED    = (0, 0, 255)
 

def imshow(image, windowname, text1, text2, color2 = COLOR_WHITE, text3 = None, color3 = None):
    """Show frame with "windowname" and 3 line of text"""
    if image is None:
        return
    if text1 is not None:
        cv.rectangle(image, (10, 10), (100,30), COLOR_WHITE, -1)
        cv.putText(image, text1, (15, 25), cv.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0))
    if text2 is not None:
        cv.rectangle(image, (10, 40), (100,60), color2, -1)
        cv.putText(image, text2, (15, 55), cv.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0))
    if text3 is not None:
        cv.rectangle(image, (10, 70), (250,90), color3, -1)
        cv.putText(image, text3, (15, 85), cv.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0))
    cv.imshow(windowname, image)


def training(capture,  frameBufcount, delay = 50, showframe = True):
    """take first 'frameBufcount' frames from 'capture' and make median background, showframe make training frames visible"""
    frames = []
    # read first frame for frame varibles 
    ret, frame = capture.read()
    bgImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    for frameN in range(frameBufcount):
        #read frame
        ret, frame = capture.read()
        if frame is None:
            return None
        #create gray image
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #update the background model
        frames.append(grayFrame)
        #imshow training frame
        if showframe:
            imshow(frame, 'Video frame', None, str(frameN), COLOR_YELLOW, "Training", COLOR_YELLOW)
        # keyboard interupt
        keyboard = cv.waitKey(delay)
        if keyboard == 'q' or keyboard == 27:
            break
    #calculate background model
    bgImage = np.median(frames, axis=0).astype(dtype=np.uint8)
    return bgImage

def main():
# Help and args
    parser = argparse.ArgumentParser(description='Test project. Detect camera cover. Exit: "ESC" or "q". Copyright km3.smtp@gmail.com')
    parser.add_argument('--video', type=str, help='cam - for webcam, or videofile name', default='cam')
    parser.add_argument('--thresholdT', type=int, help='threshold T (0..255) Background change threshold', default=50)
    parser.add_argument('--treshholdN', type=int, help='threshold N Pixel count threshold', default=20000)
    parser.add_argument('--frameBuf', type=int, help='Number of frames for training background image', default=60)
    parser.add_argument('--showBG', type=str, help='Show calculated background image', default=None)
    parser.add_argument('--showDiff', type=str, help='Show diff image', default=None)
    parser.add_argument('--delay', type=int, help='Delay between frames, ms', default=50)
    args = parser.parse_args()
# Help and args

# Open video
    # webcam by default 
    capture = cv.VideoCapture(0)
    if (args.video != 'cam'):
        # open video file
        capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.video))
    if not capture.isOpened():
        print('Cant open: ' + args.video)
        exit(0)
# Open video

# var
    trechhold = 0
    # training 
    bgImage = training(capture, args.frameBuf, args.delay)
    if bgImage is None:
        print("Video is to short !")
        return 1
    frameDiff = bgImage
    
    # "background" image
    if args.showBG is not None:
        cv.imshow('Background Image', bgImage)

# read video loop
    while True:
        #read frame
        ret, frame = capture.read()
        if frame is None:
            break
        #create gray image
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #calculate diff of current images
        frameDiff = cv.absdiff(bgImage, grayFrame)
        th, frameDiff = cv.threshold(frameDiff, args.thresholdT, 255, cv.THRESH_BINARY)

        #calculate current image treshold
        trechhold = cv.countNonZero(frameDiff)

# imshow
        if (trechhold < args.treshholdN):
            imshow(frame, 'Video frame', str(args.treshholdN), str(trechhold), COLOR_WHITE)
        else:
            imshow(frame, 'Video frame', str(args.treshholdN), str(trechhold), COLOR_RED, "!!! Camera Covered !!!", COLOR_RED)

        # Diff image (current - background)
        if args.showDiff is not None:
            cv.imshow('Diff image', frameDiff)
# imshow

# keyboard interupt
        keyboard = cv.waitKey(args.delay)
        if keyboard == 'q' or keyboard == 27:
            break
# keyboard interupt
# read video loop

# cleanup
    cv.destroyAllWindows()
    capture.release()
# cleanup
    return 0


if __name__ == "__main__":
    main()