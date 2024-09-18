import cv2, os, glob, time
import numpy as np
import parameters as para
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats
from statistics import mean
from hikvisionapi import Client
from matplotlib.backend_tools import Cursors

LINE_THRESHOLD = 20 # 直线识别的阈值

# 大致估计的范围，在此之内的直线才会被识别
# 此范围可用show_image()函数手动测算
BOUND_HER = (310, 1935) # 水平边界（X坐标）
BOUND_VER = (58, 1230) # 竖直便捷（Y坐标），下界需要非常准确

############################## 摄像头相关 ###############################

def cam_capture_frame(file_name='capture.jpg'):
    cam = Client("http://"+para.CAM_IP, para.CAM_USER, para.CAM_PWD, timeout=10)
    response = cam.Streaming.channels[102].picture(method='get', type='opaque_data')

    with open(file_name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

# 计算摄像头校准矩阵，棋盘格大小为(行数-1, 列数-1)
def cal_calibration_matrix(chessboard_size = (12, 8)):
    def get_image():
        count = 0
        while os.path.exists(f'calibration_images/cal_{count}.jpg'):
            count += 1
        cam_capture_frame(f'calibration_images/cal_{count}.jpg')
        return

    # 生成棋盘格的世界坐标系中的坐标
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # 储存所有图像的角点及其对应的世界坐标
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # 读取所有校准图像
    images = glob.glob('calibration_images/*.jpg')
    if images == []:
        raise FileNotFoundError("No calibration images in folder ./calibration_images/, please capture some images by running \"cam_calibration.get_image()\"")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # 如果找到，增加到点集合中
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # 显示角点
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # 摄像头校准
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    os.makedirs('matrix', exist_ok=True)
    # 保存在一个文件中
    np.save('matrix/calibration_mtx.npy', mtx)
    np.save('matrix/calibration_dist.npy', dist)

    return mtx, dist

def cam_calibration(mtx=None, dist=None, plot=False):
    if mtx is None or dist is None:
        mtx = np.load('matrix/calibration_mtx.npy')
        dist = np.load('matrix/calibration_dist.npy')

    # 测试校正效果
    img = cv2.imread('capture.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 畸变校正
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # 裁剪图像
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('capture_fixed.jpg', dst)

    if plot:
        cv2.imshow('calibresult', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return



############################ 视觉计算相关 #############################

def show_image(img_path='capture_fixed.jpg'):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def image_size(img_path='capture_fixed.jpg'):
    img = cv2.imread(img_path)
    return img.shape[:2]

def annotation(num=4, img_path='capture_fixed.jpg'): # 手动标注图像点
    points = []
    click_count = 0
    def onclick(event):
        nonlocal click_count
        x, y = event.xdata, event.ydata
        points.append((int(x), int(y)))
        # print(f"点击的位置: ({int(x)}, {int(y)})")
        # print(f"映射坐标: {pixel_to_table((int(x), int(y)))}")

        plt.plot(x, y, 'ro')
        plt.draw()

        click_count += 1
        if click_count >= num:
            fig.canvas.mpl_disconnect(cid)
            plt.pause(1)
            plt.close()
    
    def hover(event):
        if fig.canvas.widgetlock.locked():
            # Don't do anything if the zoom/pan tools have been enabled.
            return
        fig.canvas.set_cursor(Cursors.SELECT_REGION if event.inaxes else Cursors.POINTER)

    def escape(event):
        if event.key == 'escape':
            fig.canvas.mpl_disconnect(cid)
            plt.close()
    
    img = plt.imread(img_path)
    fig, ax = plt.subplots()
    fig.canvas.set_cursor(Cursors.SELECT_REGION) # 改为十字光标
    ax.imshow(img)
    ax.axis('off')
    # ax.imshow(img, aspect='auto', extent=ax.get_window_extent().bounds)
    # fig.canvas.manager.full_screen_toggle()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('motion_notify_event', hover)
    # fig.canvas.mpl_connect('key_press_event', escape)
    print(f"请点击图像进行标注，标注{num}个点后自动结束...")
    plt.show()
    print("点击的位置:", points)
    # plt.close()
    return points

def cal_transform_matrix(img_path='capture_fixed.jpg', method="manual" ,plot=False):
    '''
    计算像素到桌面（机械臂坐标）的映射矩阵，自动保存到'matrix/transform_M.npy'中

    参数：
    - method：设定了标定桌面边界的方式，'manual'为手动标注四个角，'opencv'为通过cv2边缘检测。
    - plot：设定是否显示最终标定的桌面框。
    '''

    if method not in ["manual", "opencv"]:
        raise ValueError("Method must be 'manual' or 'opencv'.")

    # 最小二乘法拟合直线
    def fit_line(lines):
        x_coords = []
        y_coords = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        slope, intercept, _, _, _ = stats.linregress(x_coords, y_coords)

        start_point = (int(min(x_coords)), int(min(x_coords)) * slope + intercept)
        end_point = (int(max(x_coords)), int(max(x_coords)) * slope + intercept)

        start_point = tuple(map(int, start_point))
        end_point = tuple(map(int, end_point))
        
        return start_point + end_point

        # 判断直线是否在边界内
    def is_within_bounds(x1, x2, y1, y2): 
        if x1 < BOUND_HER[0] or x2 < BOUND_HER[0]:
            return False
        if x1 > BOUND_HER[1] or x2 > BOUND_HER[1]:
            return False
        if y1 < BOUND_VER[0] or y2 < BOUND_VER[0]:
            return False
        if y1 > BOUND_VER[1] or y2 > BOUND_VER[1]:
            return False
        return True

    # 过滤属于桌子四边的直线
    def filter_table_edges(lines): 
        horizontal_lines = []
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if is_within_bounds(x1, x2, y1, y2):
                if abs(y2 - y1) < LINE_THRESHOLD:
                    horizontal_lines.append(line)
                elif abs(x2 - x1) < LINE_THRESHOLD:
                    vertical_lines.append(line)
        return horizontal_lines, vertical_lines

    # 计算两条直线的交点
    def line_intersection(line1, line2): 
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # 平行线

        px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denominator
        py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denominator
        return int(px), int(py)

    image = cv2.imread(img_path)
    line_image = np.copy(image)

    if method == "opencv":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用高斯模糊去噪，增强边缘检测效果
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # 使用Canny进行边缘检测
        edges = cv2.Canny(blurred, 15, 15)

        # 使用Hough变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        # 过滤水平和竖直的边缘
        horizontal_lines, vertical_lines = filter_table_edges(lines)

        # 在图像上绘制检测到的直线
        for line in horizontal_lines+vertical_lines: # 若最终边界有误，可查看检测到的直线
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 区分四条边的识别线
        boarders_left = []
        boarders_right = []
        boarders_top = []
        boarders_bottom = []

        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            if mean((y1, y2)) < mean(BOUND_VER): # 中心以上
                boarders_top.append(line)
            else: # 中心以下
                boarders_bottom.append(line)

        for line in vertical_lines:
            x1, y1, x2, y2 = line[0]
            if mean((x1, x2)) < mean(BOUND_HER): # 中心左侧
                boarders_left.append(line)
            else: # 中心右侧
                boarders_right.append(line)

        # 最小二乘法拟合边界直线，edge格式为[x1, y1, x2, y2]
        edge_left = fit_line(boarders_left)
        edge_right = fit_line(boarders_right)
        edge_top = fit_line(boarders_top)
        edge_bottom = fit_line(boarders_bottom)
        # print(edge_left, edge_right, edge_top, edge_bottom)
        
        cv2.line(line_image, edge_left[:2], edge_left[2:], (255, 0, 0), 2)
        cv2.line(line_image, edge_right[:2], edge_right[2:], (255, 0, 0), 2)
        cv2.line(line_image, edge_top[:2], edge_top[2:], (255, 0, 0), 2)
        cv2.line(line_image, edge_bottom[:2], edge_bottom[2:], (255, 0, 0), 2)

        # 四个顶点的位置
        top_left = line_intersection(edge_left, edge_top)
        top_right = line_intersection(edge_right, edge_top)
        bottom_left = line_intersection(edge_left, edge_bottom)
        bottom_right = line_intersection(edge_right, edge_bottom)
        corners = [top_left, top_right, bottom_right, bottom_left]

    if method == "manual":
        corners = annotation(img_path=img_path)
    
    # print("Detected rectangle corners:", corners)

    if plot:
        # 绘制桌子的四个顶点
        for corner in corners:
            cv2.circle(line_image, corner, 10, (0, 0, 255), -1)
        # 绘制四个顶点之间的连线
        for i in range(4):
            cv2.line(line_image, corners[i], corners[(i+1)%4], (0, 0, 255), 2)
        # 显示带有检测到的直线和桌子四个顶点的图像
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
        plt.show()
    
    # 像素坐标->机械臂桌面坐标
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(list(para.TABLE_CORNER.values())))
    os.makedirs('matrix', exist_ok=True)
    np.save('matrix/transform_M.npy', M)

    return M

def pixel_to_table(pixel_point):
    # 将像素点转为齐次坐标
    pixel_point = np.array([pixel_point[0], pixel_point[1], 1], dtype="float32")

    if not os.path.exists('matrix/transform_M.npy'):
        M = cal_transform_matrix('capture_fixed.jpg')
    else:
        M = np.load('matrix/transform_M.npy')

    # 通过透视变换矩阵变换为世界坐标
    world_point = np.dot(M, pixel_point)
    # 转换为实际坐标（除以齐次坐标的最后一个分量）
    world_point = world_point / world_point[2]
    return world_point[:2]  # 返回X, Y




if __name__ == "__main__":
    # cam_capture_frame()
    # cal_calibration_matrix()
    # cam_calibration(plot=True)
    # cal_transform_matrix(plot=True)
    show_image()
    # print(pixel_to_table((1145, 405)))