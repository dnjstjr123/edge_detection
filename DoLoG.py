# new idea
# Differenece of Lapalacian of Gaussian filter
# 1. 가우시안 적용 필터(이미지의 Noise를 제거)
# 2. 가우시안 필터 적용 후, Laplacian 적용 필터
# 3. 두 Laplacian의 차이 적용 필터
import numpy as np  # 자연상수와 제곱, matrix를 만들기 위해 활용한 모듈


# 1. 가우시안 필터로 이미지의 Noise를 제거한다.
def gaussian_filter(size, sigma):
    x = np.zeros((size, size))  # x의 행렬
    y = np.zeros((size, size))  # y의 행렬
    element = -(int(size) // 2)
    for i in range(size):  # x kernel에 element 넣기
        for j in range(size):
            x[i][j] += element
        element += 1
    for i in range(size):  # y kernel에 element 넣기
        element = -(element - 1)
        for j in range(size):
            y[i][j] += element
            element += 1
    #  2d 가우시안 수식에 시그마와 matrix x, y를 통한 중심에서부터의 거리 계산 값을 대입하여 생성된 필터
    result = (1 / (2 * np.pi * np.power(sigma, 2))) * np.exp(-((np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2))))

    return result


# 2. 가우시안 필터링 한 이미지를 가져와서, 라플라시안 적용
def Laplacian_of_Gaussian_filter(img, size):
    if size == 3:  # size가 3인 경우 Laplacian mask
        mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif size == 5:  # size가 5인 경우 Laplacian mask
        mask = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
    LoG = convolution(img, mask)  # 가우시안 필터링한 이미지에 라플라시안 필터 적용
    return LoG


# 3. 두 Laplacian의 차이 적용
def Difference_of_Laplacian_of_Gaussian_filter(LoG1, LoG2):
    # LoG 필터를 적용한 두 사진에 대한 차이를 적용
    x, y = np.shape(LoG1)
    DoLoG = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            DoLoG[i][j] = float(LoG1[i][j]) - float(LoG2[i][j])

    return DoLoG


# padding 과정 진행
def Padding(img, size):
    x, y = img.shape  # 필터링된 이미지의 (행, 열)
    padding_size = size // 2  # 패딩의 사이즈
    result = np.zeros((x + 2 * padding_size, y + 2 * padding_size))  # 패딩 과정을 진행한 결과 행렬
    if padding_size == 0:  # 이미지의 경계선을 넘지 않는 경우
        result = img.copy()
    else:  # 이미지의 경계선을 넘는 경우
        result[padding_size:-(padding_size), padding_size:-(padding_size)] = img.copy()  # 패딩 적용
    return result


# convolution 과정
def convolution(img, filter, stride=1):  # (이미지 array, filter array, 필터의 이동 간격)
    result = []  # 필터링된 이미지 배열
    img_x, img_y = np.shape(img)  # 원본의 (행, 열)
    filter_x, filter_y = np.shape(filter)  # 필터의 (행, 열)
    for i in range(0, img_x-filter_x + 1, stride):  #  Convolution 과정
        for j in range(0, img_y-filter_y + 1, stride):
            result.append((img[i:i+filter_x, j:j+filter_y]*filter).sum())  # 범위를 지정하고 fiter와 곱해준다.
    result_x = int((img_x-filter_x)/stride)+1
    result_y = int((img_y-filter_y)/stride)+1
    result = np.array(result).reshape(result_x, result_y)
    result = np.array(result, dtype=np.uint8)  # uint8로 행렬 요소 변경

    return result


import cv2  # 이미지를 불러오고 보여주기 위해 사용

img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)  # 컬러 이미지를 불러와서 흑백으로 변경

size = 3  # 필터의 사이즈는 = 3으로 둔다.

gaussian_1 = convolution(img, gaussian_filter(size, 1))  # size = 3, sigma = 1인 gaussian filter 적용
gaussian_1 = Padding(gaussian_1, size)  # 패딩 적용
LoG_1 = Laplacian_of_Gaussian_filter(gaussian_1, size)  # 가우시안 필터가 적용된 이미지에 라플라시안 필터 적용

gaussian_2 = convolution(img, gaussian_filter(size, 0.5))  # size = 3, sigma = 0.5인 gaussian filter 적용
gaussian_2 = Padding(gaussian_2, size)  # 패딩 적용
LoG_2 = Laplacian_of_Gaussian_filter(gaussian_2, size)  # 가우시안 필터가 적용된 이미지에 라플라시안 필터 적용

DoLoG = Difference_of_Laplacian_of_Gaussian_filter(LoG_1, LoG_2)  # sigma 값이 다른 두 LoG가 적용된 이미지의 차를 이용한 필터 적용

cv2.imshow('DoLoG', DoLoG)  # DoLoG가 적용된 이미지 보여주기
cv2.waitKey(0)
cv2.destroyAllWindows()
