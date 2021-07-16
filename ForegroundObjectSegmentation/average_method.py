import numpy as np
import cv2

# Pedestrians
# ROI_path = 'pedestrians/temporalROI.txt'
# image_path = 'pedestrians/input'
# groundtruth_path = 'pedestrians/groundtruth'
# Highway
# ROI_path = 'highway/temporalROI.txt'
# image_path = 'highway/input'
# groundtruth_path = 'highway/groundtruth'
# Office
ROI_path = 'office/temporalROI.txt'
image_path = 'office/input'
groundtruth_path = 'office/groundtruth'

f = open(ROI_path, 'r')
line = f.readline()
ROI_start, ROI_end = line.split()
ROI_start = int(ROI_start)
ROI_end = int(ROI_end)

ANALYSIS_STEP = 1
TP = 0
FN = 0
FP = 0

N = 60
XX = 360  # highway: 320
YY = 240
BUF = np.zeros((YY, XX, N), np.uint8)
iN = 0

initial_image = cv2.imread(f'{image_path}/in%06d.jpg' % 300)
initial_image_gray = cv2.cvtColor(src=initial_image, code=cv2.COLOR_BGR2GRAY)

for i in range(ROI_start, ROI_end, ANALYSIS_STEP):  # start, stop, step
    image = cv2.imread(f'{image_path}/in%06d.jpg' % i)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    BUF[:, :, iN] = image_gray
    iN += 1
    if iN == 60:
        iN = 0

    image_mean = np.uint8(np.mean(BUF, 2))
    absolute_difference = cv2.absdiff(image_gray, image_mean)
    binarization = cv2.threshold(src=absolute_difference, thresh=15, maxval=255, type=cv2.THRESH_BINARY)
    binarized_image = binarization[1]
    operation_kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binarized_image, operation_kernel, 1)
    dilated_image = cv2.dilate(eroded_image, operation_kernel, 1)
    # blurred_image = cv2.medianBlur(dilated_image, 5)
    median_kernel = 3
    blurred_image = cv2.medianBlur(dilated_image, median_kernel)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(blurred_image)
    cv2.imshow("Labels", np.uint8(labels / stats.shape[0] * 255))

    image_vis = image
    if stats.shape[0] > 1:
        tab = stats[1:, 4]
        pi = np.argmax(tab)
        pi = pi + 1
        cv2.rectangle(image_vis, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0] + stats[pi, 2], stats[pi, 1] + stats[pi, 3]),
                      (255, 0, 0), 2)
        cv2.putText(image_vis, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.putText(image_vis, "%d" % pi, (int(centroids[pi, 0]), int(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0))

    cv2.imshow("Labels", image_vis)
    cv2.waitKey(10)

    groundtruth = cv2.imread(f'{groundtruth_path}/gt%06d.png' % i)
    groundtruth_gray = cv2.cvtColor(groundtruth, cv2.COLOR_BGR2GRAY)

    TP_M = np.logical_and((blurred_image == 255), (groundtruth_gray == 255))
    FN_M = np.logical_and((blurred_image == 0), (groundtruth_gray == 255))
    FP_M = np.logical_and((blurred_image == 255), (groundtruth_gray != 255))

    TP_S = np.sum(TP_M)
    FN_S = np.sum(FN_M)
    FP_S = np.sum(FP_M)

    TP = TP + TP_S
    FN = FN + FN_S
    FP = FP + FP_S

P = TP / (TP + FP)
R = TP / (TP + FN)
F1 = 2*P*R / (P + R)

print('R parameter value:', R)
print('P parameter value:', P)
print('F1 parameter value:', F1)
