## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
## Pipeline

---
### Camera calibration
First, I'll compute the camera calibration using chessboard images. Then I will get the camera parameters which will be save to "CamParams-0.pkl" for later use. Using `CamParams.load()` function not only avoids repetitive calibration but also accelerates pre-process speed.

Change the line `if 0:` to `if 1:` if you want switch to calibration mode from load mode.
> `keep_global_var` and `printvars` is used to remove and show global defined variables.

```{.python .input}
%reset -f
import numpy as np
import cv2
import glob
import matplotlib, matplotlib.pyplot as plt
%reload_ext autoreload
%aimport util
%autoreload 1 
%matplotlib inline

np.set_printoptions(2)

def printvars():
    tmp = globals().copy()
    print('all defined vars:', end=' ') 
    [print(k, end=', ') for k, v in tmp.items() if not str(type(v)).endswith('module\'>') 
                                                  and not k.startswith('_') 
                                                  and k != 'tmp' and k != 'In' and k != 'Out' 
                                                  and not hasattr(v, '__call__')]

def keep_global_var(var_list):
    tmp = globals().copy()
    v = [k for k, v in tmp.items() if not k.startswith('_') 
                                      and not str(type(v)).endswith('module\'>') 
                                      and k != 'tmp' and k != 'In' and k != 'Out' 
                                      and not hasattr(v, '__call__')]
    for _v in v:
        if _v not in var_list:
            del globals()[_v]
    

# Make a list of calibration images
camParams = util.CamParams()

if 0:
    images = glob.glob('../camera_cal/calibration*.jpg')
    camParams.calibration(images, 6, 9)
    camParams.save(0)
else:
    camParams.load(0)
    

mtx, dist, _, _ = camParams.get()
img_dist = matplotlib.image.imread('../camera_cal/calibration1.jpg')
img_undist = cv2.undistort(img_dist, mtx, dist, None, mtx)

util.plot_comparison(img_dist, img_undist)

# Only keep `camParams` in the python environment
keep_global_var(['camParams']); printvars()
```

### Camera Undistortion

Run below cell to apply a distortion correction to raw images.
> Notice that this step has been writen into `util.LaneDetector.preProcess()` and will be automatically invoked when `preProcess` runs.

```{.python .input}
#fn_img = '../test_images/straight_lines1.jpg'
fn_img = '../test_images/test3.jpg'

# Distorted Image
img_rgb_dist = matplotlib.image.imread(fn_img)

img_rgb = camParams.undistort(img_rgb_dist)

# Convert to Gary for later use
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

util.plot_comparison(img_rgb_dist, img_rgb)

keep_global_var(['camParams'])
```

---
### Perpective Transform
Four source points of a quadrangle should be identified for a perspective transform. In order to pick four points in a trapzoidal shape representing a rectangle I have to find a pair of parallel lane lines as straight as possible.

I've provided `persTrans.get_y_from_x` and `persTrans.get_x_from_y` to help get the coordinates of the four points in the image coordinate. Fine-tune `pts1` and `pts2` to obtain the best perpective transform.

> Don't forget to update `pts1` and `pts2` to `util.LaneDetector.loadCamParams()` after adjustment.

```{.python .input}
pers_img = camParams.undistort(matplotlib.image.imread('../test_images/straight_lines1.jpg'))
persTrans = util.PersTrans()

pts1 = np.float32([[204, 720], [1110, 720], [503, 515], [789, 515]])
pts2 = np.float32([[204, 720], [1110, 720], [204, 300], [1110, 300]])

print('y: %.1f'%persTrans.get_y_from_x(550, pts1[2], pts1[0]))
print('x: %.1f'%persTrans.get_x_from_y(483, pts1[1], pts1[3]))

# 源坐标，4个蓝色点
pts1 = np.float32([[204, 720], [1110, 720], [550, 483], [738, 483]])
# 转换后坐标，4个绿色小圈
pts2 = np.float32([[300, 720], [940, 720], [300, 0], [940, 0]])

# Compute perpective transform
pers_M = persTrans.computeM(pts1, pts2, pers_img)

# Apply perpective transform
# `img.shape` is (rows, cols), here should be (cols, rows)
pers_img_dst = cv2.warpPerspective(pers_img, pers_M, (1280, 720))

util.plot_comparison(pers_img, pers_img_dst)

keep_global_var(['camParams'])
```

Try some other points to see which one is most suitable for using in the pipeline.

```{.python .input}
fn_img_list = images = glob.glob('../test_images/my*')
fn_img_list.insert(0, '../test_images/straight_lines1.jpg')
persTrans = util.PersTrans()

"""
pts[0], 源坐标， 4个蓝色点
pts[1], 转换后坐标， 4个绿色小圈
pts[2], (rows, cols)
"""
pts_list = [[np.float32([[204, 720], [1110, 720], [503, 515], [789, 515]]),
             np.float32([[204, 720], [1110, 720], [204, 300], [1110, 300]]),
             (720, 1280)],
            [np.float32([[204, 720], [1110, 720], [550, 483], [738, 483]]),
             np.float32([[300, 720], [940, 720], [300, 0], [940, 0]]),
             (720, 1280)],
            [np.float32([[204, 720], [1110, 720], [550, 483], [738, 483]]),
             np.float32([[320, 820], [900, 820], [320, 0], [900, 0]]),
             (820, 1280)]]

for fn_img in fn_img_list:
    img = camParams.undistort(matplotlib.image.imread(fn_img))
    img_disp_list = [img]
    
    for pts in pts_list:
        # Compute perpective transform
        pers_M = persTrans.computeM(pts[0], pts[1])
    
        # Apply perpective transform
        # param shape `pts[2]` is (rows, cols), here should be (cols, rows)
        img_pers = cv2.warpPerspective(img, pers_M, pts[2][1::-1])
        img_disp_list.append(img_pers)
        
    util.plot_stack(img_disp_list, titles=[fn_img])
    
keep_global_var(['camParams'])
```

---
### Edge Detection
I've tried the edge detection method discussed in the class, and also changed the order of edge detection and perspective tranform to find which order is better. The comparison shows that the order of which one going first has little effect on the final result. Overall， edge detection going first will results in fewer noise, so I will do perspective transform after edge detection.

However, from the result shown after running below cell, we can see that the original color and gradient algorithm does not work well under shadow environment such as "myest0.png" and "mytest3.png".

```{.python .input}
fn_img_list = ['../test_images/mytest0.png',
               '../test_images/mytest1.png',
               '../test_images/mytest3.png']

#fn_img_list.insert(0, '../test_images/straight_lines1.jpg')
#fn_img_list.insert(0, '../test_images/test2.jpg')

from util import EdgeDetector
laneDetector = util.LaneDetector()
laneDetector.loadCamParams(0)

def detPossibleEdges(img_rgb, sob_x_thresh=(20, 100), sat_thresh=(170, 255), ret_temps=False):
    """
    Detect Possible Edges
    """
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    l_channel, s_channel = hls[:, :, 1], hls[:, :, 2]
    
    # Sobel x
    sobelx_l = EdgeDetector.Sobel(l_channel, 1, 0)

    # Threshold x gradient
    sob_x_binary_l = EdgeDetector.BinaryThresholding(sobelx_l, sob_x_thresh)

    # Threshold color channel
    sat_binary = EdgeDetector.BinaryThresholding(s_channel, sat_thresh)

    if ret_temps == True:
        temps = {'l': l_channel, 
                 'lx': sobelx_l, 
                 'lb': sob_x_binary_l, 
                 's': s_channel, 
                 'sb': sat_binary}
        return sob_x_binary_l | sat_binary, temps
    else:
        return sob_x_binary_l | sat_binary
    

for fn_img in fn_img_list:
    # Read and undistort the image
    img_rgb = cv2.cvtColor(cv2.imread(fn_img), cv2.COLOR_BGR2RGB)
    img_rgb = laneDetector.preProcess(img_rgb)

    # 全图Edge => Pers
    img_edge, tmp = detPossibleEdges(img_rgb, ret_temps=True)

    img_edge_pers = cv2.warpPerspective(img_edge, laneDetector.pers_M, laneDetector.pers_shape[1::-1])

    temp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,3));
    img_edge_pers_denoised = cv2.morphologyEx(img_edge_pers, cv2.MORPH_OPEN, temp);
    
    # Pers => Edge
    #img_pers = cv2.warpPerspective(img_rgb, laneDetector.pers_M, laneDetector.pers_shape[1::-1])
    #img_pers_edge = laneDetector.detPossibleEdges(img_pers, ret_temps=True)
    #img_pers_edge_denoised = cv2.morphologyEx(img_pers_edge, cv2.MORPH_OPEN, temp);

    # Plot the result
    util.plot_stack([tmp['l'], tmp['lx'], tmp['lb']], titles=['%s: l'%fn_img, 'lx', 'lb'])
    util.plot_stack([tmp['s'], tmp['sb'], img_edge_pers_denoised], titles=['s', 'sb', 'img_edge_pers'])
    #util.plot_stack([img_edge, img_edge_pers, img_edge_pers_denoised], 
    #                titles=['img_edge_roi', 'img_edge_pers', 'img_edge_pers_denoised'])
    
keep_global_var([])
```

To make the code can work well in dark environments, instead of thresholding `s_channel` I apply an sobel along horizontal direction on `s_channel` which improve a lot on edge detection. You can pay special attention to "myest0.png" and "mytest3.png".

```{.python .input}
fn_img_list = images = glob.glob('../test_images/my*')
fn_img_list.insert(0, '../test_images/straight_lines1.jpg')
fn_img_list.insert(1, '../test_images/test1.jpg')

from util import EdgeDetector, LaneDetector
laneDetector = util.LaneDetector()
laneDetector.loadCamParams(0)

def detPossibleEdges(img_rgb, sob_x_thresh=(20, 100), ret_temps=False):
    """
    Detect Possible Edges
    """
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    l_channel, s_channel = hls[:, :, 1], hls[:, :, 2]
    
    # Sobel x
    sobelx_l = EdgeDetector.Sobel(l_channel, 1, 0)

    # Threshold x gradient
    sob_x_binary_l = EdgeDetector.BinaryThresholding(sobelx_l, sob_x_thresh)

    # Threshold color channel    
    # Sobel x on s_channel
    sobelx_s = EdgeDetector.Sobel(s_channel, 1, 0)

    # Threshold x gradient
    sob_x_binary_s = EdgeDetector.BinaryThresholding(sobelx_s, sob_x_thresh)    
    
    if ret_temps == True:
        temps = {'l': l_channel, 
                 'lx': sobelx_l, 
                 'lb': sob_x_binary_l, 
                 's': s_channel, 
                 'sx': sobelx_s,
                 'sb': sob_x_binary_s}
        return sob_x_binary_l | sob_x_binary_s, temps
    else:
        return sob_x_binary_l | sob_x_binary_s


for fn_img in fn_img_list:
    # Read and undistort the image
    img_rgb = cv2.cvtColor(cv2.imread(fn_img), cv2.COLOR_BGR2RGB)
    img_rgb = laneDetector.preProcess(img_rgb)

    # 全图Edge => Pers
    img_edge, tmp = detPossibleEdges(img_rgb, ret_temps=True)
    LaneDetector.KeepROI(img_edge)

    img_edge_pers = cv2.warpPerspective(img_edge, laneDetector.pers_M, laneDetector.pers_shape[1::-1])

    temp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,3));
    img_edge_pers_denoised = cv2.morphologyEx(img_edge_pers, cv2.MORPH_OPEN, temp);
    
    # Pers => Edge
#     img_pers = cv2.warpPerspective(img_rgb, laneDetector.pers_M, laneDetector.pers_shape[1::-1])
#     img_pers_edge = laneDetector.detPossibleEdges(img_pers)
#     img_pers_edge_denoised = cv2.morphologyEx(img_pers_edge, cv2.MORPH_OPEN, temp);
#     util.plot_stack([img_pers, img_pers_edge, img_pers_edge_denoised], 
#                     titles=['img_pers', 'img_pers_edge', 'img_pers_edge_denoised'])
    
    # Plot the result
    util.plot_stack([tmp['l'], tmp['lx'], tmp['lb']], titles=['%s: l'%fn_img, 'lx', 'lb'])
    util.plot_stack([tmp['s'], tmp['sx'], tmp['sb']], titles=['s', 'sx', 'sb'])
    util.plot_stack([img_edge, img_edge_pers, img_edge_pers_denoised], 
                    titles=['img_edge_roi', 'img_edge_pers', 'img_edge_pers_denoised'])

keep_global_var([])
```

---
### Finding the Lines: 

---
#### Histogram Peaks
There is no big change here with class solution, but I take the edge width into account when I compute the histogram. Instead of summing up all pixels along each column, I sum up the pixels inside adjacent columns which I set the width to 40, see `DetEdgePeaks(binary_edges, edge_width=40)`. Thus, many small edges will be dropped as a lane line should have a certain width.

```{.python .input}
from util import EdgeDetector, LaneDetector

fn_img_list = images = glob.glob('../test_images/*')
#fn_img_list.insert(0, '../test_images/straight_lines1.jpg')
#fn_img_list.append('../test_images/test1.jpg')

laneDetector = LaneDetector()
laneDetector.loadCamParams(0)


def PreProcess(fn_img):
    # Read and undistort the image
    img_rgb = cv2.cvtColor(cv2.imread(fn_img), cv2.COLOR_BGR2RGB)
    img_rgb = laneDetector.preProcess(img_rgb)

    # 全图Edge => Pers
    img_edge = LaneDetector.detPossibleEdges(img_rgb)
    LaneDetector.KeepROI(img_edge)

    img_edge_pers = cv2.warpPerspective(img_edge, laneDetector.pers_M, laneDetector.pers_shape[1::-1])

    temp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,3));
    img_edge_pers_denoised = cv2.morphologyEx(img_edge_pers, cv2.MORPH_OPEN, temp);
    
    return img_edge_pers_denoised


def DetEdgePeaks(binary_edges, edge_width=40):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_edges[binary_edges.shape[0]//2:,:], axis=0)
    #histogram = np.sum(binary_edges, axis=0)

    # Compute histogram within continuous columns, determined by `edge_width`
    histogram = np.array([np.sum(histogram[i:i+edge_width])
                          for i in range(len(histogram))])
    for i in reversed(range(1, edge_width)):
        histogram[-i] += histogram[-1]*(edge_width-i)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    return leftx_base, rightx_base, histogram/edge_width


for fn_img in fn_img_list:
    bimg_edge = PreProcess(fn_img)
    
    # Create histogram of image binary activations
    leftx_base, rightx_base, histogram = DetEdgePeaks(bimg_edge)

    leftxy, rightxy, out_img = LaneDetector.FindLanePixels(bimg_edge, leftx_base, rightx_base)
    
    # Plot the result
    util.plot_stack([bimg_edge, out_img, out_img])
    # Visualize the resulting histogram
    plt.cla(); plt.plot(histogram); 
    axes = plt.gca(); axes.set_xlim([0, 1280]); axes.set_ylim([0, 720])

keep_global_var([])
```

#### Sliding windows and Fit Polynomial
As shown in the previous animation, I have got two highest peaks from the histogram as a starting point for the determining where the lane lines are. Now I will use sliding windows moving upward in the image to determine where the lane lines go. Notice that in my solution the movement trend of sliding windows is used to predict the center of new sliding window, which is very important to track a curved lane line.

In order to save time and utilize the characteristics that lane lines appear at a close position on the two frames, a margin around the previous line position is the first target to search. **However, there is the necessity for checking similarity and confidence of previous lane lines, and switch back to whole region resarch if fail to meet the requirement.**

---
### Measuring Curvature
$x = A'y^2 + B'y + C$

o 代表 offset，由于计算 erpective transform 时候引起的

$m(x+o) = A(ny)^2 + B(ny) + C$  
$mx = (An^2)y^2 + (Bn)y + (C-mo)$  
$x = (An^2/m)y^2 + (Bn/m)y + (C/m-o)$  


$$\begin{cases}
An^2/m = A' \\
Bn/m = B' \\
C/m - o = C'\\
\end{cases}$$

进一步得到
  
$$\begin{cases}
A = A'm/n^2 \\
B = B'm/n \\
C = C'm + mo\\
\end{cases}$$


---
## Whole Processing of an Image or a video

```{.python .input  n=6}
%reset -f
import numpy as np
import cv2
import glob
import matplotlib, matplotlib.pyplot as plt
%reload_ext autoreload
%aimport util
%autoreload 1 
%matplotlib inline
np.set_printoptions(2)

LaneDetector = util.LaneDetector # Equal to "from util import LaneDetector"

laneDetector = LaneDetector()
laneDetector.loadCamParams(0)

def Overlay_Lane(img_src, coef_l, coef_r, pers_shape, pers_M_inv, beta=0.2, alpha=1):
    """
    @alpha – weight of the `img_src` array elements.
    @beta – weight of the lane layer array elements.
    """
    ploty = np.linspace(0, pers_shape[0]-1, pers_shape[0])
    left_fitx = coef_l[0]*ploty**2 + coef_l[1]*ploty + coef_l[2] if coef_l != None else []
    right_fitx = coef_r[0]*ploty**2 + coef_r[1]*ploty + coef_r[2]  if coef_r != None else []
    
    pers_shape = list(pers_shape); pers_shape.append(3)
    img_lane = np.zeros(pers_shape, img_src.dtype)
    
    if len(left_fitx) == len(right_fitx):
        for xl, xr, y in zip(left_fitx.astype(np.int), 
                             right_fitx.astype(np.int), 
                             ploty.astype(np.int)):
            cv2.line(img_lane, (xl, y), (xr, y), (0, 255, 0), 1)

    img_lane_warpinv = cv2.warpPerspective(img_lane, pers_M_inv, img_src.shape[1::-1])
    
    #util.plot_stack(img_lane_warpinv)
    
    return cv2.addWeighted(img_src, alpha, img_lane_warpinv, beta, gamma=0)
     

def FrameProcess(img_rgb, track_similarity=90, track_conf=50, show_model=0):
    return_slidewindows = True if show_model == 1 else False
    
    # Apply undistortion
    img_rgb = laneDetector.preProcess(img_rgb)

    binary_edges = LaneDetector.detPossibleEdges(img_rgb)
    LaneDetector.KeepROI(binary_edges)
  
    # Apply perpective transform, `img.shape` lies as (rows, cols), here should be (cols, rows)
    bin_edge_pers = cv2.warpPerspective(binary_edges, laneDetector.pers_M, (1280, 720)) 
    
    temp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,3));
    bin_edge_pers_denoised = cv2.morphologyEx(bin_edge_pers, cv2.MORPH_OPEN, temp);
    
    # Try to use the result from last frame
    left_fit, right_fit = laneDetector.getLaneHistory()
    conf, similarity = laneDetector.ComputeConf(left_fit, right_fit)
    
    if (similarity < track_similarity) and (conf > track_conf):
        left_xy, right_xy, img_lane_windows = laneDetector.ReloadLanePixels(bin_edge_pers_denoised, 
                                                                            left_fit,
                                                                            right_fit,
                                                                            return_result=return_slidewindows)
        left_fit, right_fit = LaneDetector.FitPolynomial(left_xy, right_xy)
    else:
        left_fit, right_fit = None, None
        
    # Check if need to research again
    if left_fit == None and right_fit == None:
        #print('failed')
        leftx_base, rightx_base = LaneDetector.detEdgePeaks(bin_edge_pers_denoised, edge_width=40)
        # Find our lane pixels first
        left_xy, right_xy, img_lane_windows = LaneDetector.FindLanePixels(bin_edge_pers_denoised, 
                                                                          leftx_base,
                                                                          rightx_base,
                                                                          return_result=return_slidewindows)
        left_fit, right_fit = LaneDetector.FitPolynomial(left_xy, right_xy)
        
    laneDetector.pushLane([left_fit, right_fit])
    img_overlay = Overlay_Lane(img_rgb, left_fit, right_fit, laneDetector.pers_shape, laneDetector.pers_M_inv)
    
#     tmp_left_xy = left_xy.copy() 
#     tmp_left_xy[0] = np.array(tmp_left_xy[0]) - 96
#     tmp_right_xy = right_xy.copy() 
#     tmp_right_xy[0] = np.array(tmp_right_xy[0]) + 170
    
#     tmp_left_fit, tmp_right_fit = LaneDetector.FitPolynomial(tmp_left_xy, tmp_right_xy)
    left_curverad, right_curverad = LaneDetector.MeasureCurvature(left_fit, right_fit,
                                                                  bin_edge_pers_denoised.shape[0]-1)
    center_offset = LaneDetector.MeausreVehicelCenter(left_fit, right_fit, bin_edge_pers_denoised.shape)
    
    cv2.putText(img_overlay, "radius: %dm"%(int(left_curverad)), (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 128, 0], thickness=3)
    cv2.putText(img_overlay, "offset: %.2fm"%(center_offset), (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 128, 0], thickness=3)  
    if show_model == 1:
        L1_1 = cv2.resize(img_overlay, None, fx=0.5, fy=0.5)
        conf, similarity = laneDetector.ComputeConf(left_fit, right_fit)
        cv2.putText(L1_1, "s, c: %d, %d"%(int(similarity), int(conf)), (350, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 128, 0], thickness=3)
        L1_2 = cv2.cvtColor(cv2.resize(binary_edges*255, None, fx=0.5, fy=0.5), cv2.COLOR_GRAY2RGB)
        L2_1 = cv2.cvtColor(cv2.resize(bin_edge_pers*255, None, fx=0.5, fy=0.5), cv2.COLOR_GRAY2RGB)
        L2_2 = cv2.resize(img_lane_windows, None, fx=0.5, fy=0.5)

        img_ret = np.vstack((np.hstack((L1_1, L1_2)), np.hstack((L2_1, L2_2))))
    else:
        img_ret = img_overlay
    return img_ret 


fn_img = '../test_images/test1.jpg'
img_rgb = cv2.cvtColor(cv2.imread(fn_img), cv2.COLOR_BGR2RGB)

%time img_res = FrameProcess(img_rgb, track_similarity=90, track_conf=50, show_model=1)
util.plot_stack(img_res)

%time img_res = FrameProcess(img_rgb, track_similarity=90, track_conf=50, show_model=1)
util.plot_stack(img_res)
```

```{.python .input  n=8}
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def callback(img):
    img = FrameProcess(img, track_similarity=90, track_conf=50, show_model=1)
    return img
   
input_video = "../project_video.mp4" ; white_output = 'output/0.mp4'; 
#input_video = "../challenge_video.mp4" ; white_output = 'output/1.mp4'; 
#input_video = "../harder_challenge_video.mp4" ; white_output = './output/2.mp4'; 

# To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# To do so add .subclip(start_second,end_second) to the end of the line below
# Where start_second and end_second are integer values representing the start and end of the subclip
# You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip(input_video)
# NOTE: this function expects color images!!
white_clip = clip1.fl_image(callback).subclip(0, 5)
#white_clip = clip1.fl_image(callback)


%time white_clip.write_videofile(white_output, audio=False)
```

```{.python .input  n=9}
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```

```{.python .input}
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
import os

def callback(img):
    img = FrameProcess(img, track_similarity=90, track_conf=50, show_model=0)
    return img
   
input_videos = []
input_videos.append("../project_video.mp4" )
input_videos.append("../challenge_video.mp4")
input_videos.append("../harder_challenge_video.mp4")


for input_video in input_videos:
    clip = VideoFileClip(input_video)
    white_clip = clip.fl_image(callback)
    base = os.path.basename(input_video)
    white_clip.write_videofile('output/'+base, audio=False)
```
