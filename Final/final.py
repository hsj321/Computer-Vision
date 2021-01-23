# Python 3.7
import cv2

# 출처 https://www.youtube.com/watch?v=EDT0vHsMy34&list=PL6Yc5OUgcoTmTGACTa__vnifNA744Cz-q&index=29

# 출처 ++ https://www.youtube.com/watch?v=nkzQXc0YfRA

drew = False
drawing = False
point1 = ()
point2 = ()
ymn, xmn, ymx, xmx, temp = -1, -1, -1, -1, -1

list_color = [(255,0,0), (0,255,0), (0,0,255)]
list_name = ["blue", "green", "red"]
list_count = [0,0,0]

append_count = 0
count_now = 0
count_past = 0


frameName = 'final.mp4'


def ORB(roi):
    img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_orb_1500, img_orb = None, None

    orb = cv2.ORB_create(scaleFactor=1.3, edgeThreshold=16, fastThreshold=10, nfeatures=1000)
    kp_orb, descriptors = orb.detectAndCompute(img_gray, None)
    img_orb = cv2.drawKeypoints(img_gray, kp_orb, img_orb)

    return kp_orb, descriptors





def mouse_drawing(event, x, y, flags, params):

    global point1, point2, drawing, drew, ymn, xmn, ymx, xmx, count_past, count_now

    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing is False:
            print("First Left click")
            drawing = True
            drew = False
            point1 = (x, y)
            ymn, xmn = y, x
        else:
            print("Second Left click")
            drawing = False
            drew = True
            count_past = count_now
            count_now += 1

        #print(x, y)
        #print("Left click")

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            # print("Mouse move")
            point2 = (x, y)
            ymx, xmx = y, x


# 정지된 모델 영상에서
video = cv2.VideoCapture(frameName)
_, img = video.read()
#img[img.shape[0] - image.shape[0]:img.shape[0], img.shape[1] - image.shape[1]:img.shape[1]] = image
list_roi = []
list_des = []
list_kp = []
# cap = cv2.VideoCapture(0)
while True:
    # 마우스를 드래그하여
    cv2.namedWindow("Frame")

    cv2.setMouseCallback("Frame", mouse_drawing)
    # circles = []
    roi = img

    while True:
        # 객체의 범위를 지정
        if point1 and point2:
            video = cv2.VideoCapture(frameName)
            _, img = video.read()
            #img[img.shape[0] - image.shape[0]:img.shape[0], img.shape[1] - image.shape[1]:img.shape[1]] = image
            cv2.rectangle(img, point1, point2, (0, 255, 0))
            if drew:
                if ymn > ymx :
                    temp = ymn
                    ymn = ymx
                    ymx = temp
                if xmn > xmx :
                    temp = xmn
                    xmn = xmx
                    xmx = temp
                roi = img[ymn + 1: ymx - 1, xmn + 1: xmx - 1]
                cv2.imshow("roi", roi)
                # 특징점 추출하여
                kp_opb, des = ORB(roi)
                # 특징 기술자 생성

                key = cv2.waitKey(1)
                if key == 13:
                    print("key", key)
                    print("past", count_past)
                    print("now", count_now)
                    if count_past == count_now - 1:
                        append_count += 1
                        print('append ', append_count)
                        list_roi.append(roi)
                        list_kp.append(kp_opb)
                        list_des.append(des)
                if key == 32:
                    print("key", key)
                    if count_past == count_now - 1:
                        append_count += 1
                        print('append ', append_count)
                        list_roi.append(roi)
                        list_kp.append(kp_opb)
                        list_des.append(des)
                    cv2.destroyAllWindows()
                    break


        cv2.imshow("Frame", img)

        key = cv2.waitKey(1)
        if key == 32:
            cv2.destroyAllWindows()
            break

    cv2.imshow("Frame", roi)
    key = cv2.waitKey(0)
    if key == 32:
        cv2.destroyAllWindows()
        break

print(len(list_kp))

def drawCircle(frame1, des, kp_orb, roi):
    kp, des_frame = ORB(frame1)

    if len(kp) < 15:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    #추출한 특징점이 없을 때 False반환

    matches = bf.match(des, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)


    #각 매칭점들의 좌표를 구하는 부분(지금은 필요없음)
    list_kp1 = []
    list_kp2 = []
    # For each match...
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp_orb[img1_idx].pt
        (x2, y2) = kp[img2_idx].pt

        # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))

    distance1 = ((list_kp1[0][0] - list_kp1[1][0])**2) + ((list_kp1[0][1] - list_kp1[1][1])**2)
    distance2 = ((list_kp1[1][0] - list_kp1[2][0])**2) + ((list_kp1[1][1] - list_kp1[2][1])**2)
    distance3 = ((list_kp2[0][0] - list_kp2[1][0]) ** 2) + ((list_kp2[0][1] - list_kp2[1][1]) ** 2)
    distance4 = ((list_kp2[1][0] - list_kp2[2][0]) ** 2) + ((list_kp2[1][1] - list_kp2[2][1]) ** 2)


    dist1 = 0
    dist2 = 0
    if distance1 > distance3 and distance3 != 0:
        dist1 = distance1 / distance3
    elif distance1 != 0:
        dist1 = distance3 / distance1
    if distance2 > distance4 and distance4 != 0:
        dist2 = distance2 / distance4
    elif distance2 != 0:
        dist2 = distance4 / distance2

    length = 0
    if dist1 > dist2:
        length = dist1 - dist2
    elif dist1 < dist2:
        length = dist2 - dist1


    if length > 1 or dist1 == 0 or dist2 == 0:
        print("False")
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des, des_frame, k=2)

    good = []
    # 쓸모 있는 매칭만 사용
    for match1, match2 in matches :
        if match1.distance < 0.65 * match2.distance :
            good.append([match1])

    print('matches:{}/{}'.format(len(good), len(matches)))

    if len(good) < 40:
        return False

    return True


frame_count = 0

while True:

    _, frame = video.read()
    frame_count += 1
    if frame_count % 9 != 0:
        continue
    #frame[frame.shape[0] - image.shape[0]:frame.shape[0], frame.shape[1] - image.shape[1]:frame.shape[1]] = image
    if frame is not None :

        frame_result = frame.copy()
        list_count = [0, 0, 0]
        string_count = ""

        # 컨투어로 피카츄들 구분짓기
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        ret, frame_binary = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        frame_binary = cv2.morphologyEx(frame_binary, cv2.MORPH_OPEN, kernel)
        frame_binary = cv2.morphologyEx(frame_binary, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #print("Number of contours = ", str(len(contours)))



        for contour in contours:  # 컨투어별로 특징점 매칭
            string_count = ""
            frame = cv2.drawContours(frame, [contour], 0, (255, 0, 0), 3)  # blue
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            xi, y, w, h = cv2.boundingRect(contour)

            area = cv2.contourArea(contour)

            if area < 100:
                continue
            if area > 100000:
                continue
            frame_contour = frame[y: y + h, xi: xi + w]


            # 특징점 검출 및 매칭
            for i in range(0, len(list_kp)):
                #특징점 검출 및 검출 성공 시 True반환하는 함수
                if drawCircle(frame_contour, list_des[i], list_kp[i], list_roi[i]):
                    cv2.circle(frame_result, (cx, cy), 10, list_color[i], -1)
                    cv2.putText(frame_result, list_name[i], (cx - 15, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, list_color[i], 1, cv2.LINE_AA)
                    list_count[i] += 1
                    break
        for i in range(0, len(list_kp)):
            string_each = list_name[i] + " : " + str(list_count[i]) + "  "
            string_count = string_count + string_each
        cv2.putText(frame_result, string_count, (frame_result.shape[0], 50), cv2.FONT_ITALIC, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("a", frame_result)

        key = cv2.waitKey(10)
        if key == 32:
            break
    else :
        break

video.release()

cv2.destroyAllWindows()