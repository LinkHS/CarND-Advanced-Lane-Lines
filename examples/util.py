import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(ori_img, res_img, ori_tit='Original', res_tit='Result', fontsize=30, ori_cmap='gray', res_cmap='gray'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))
    f.tight_layout()

    ax1.imshow(ori_img, cmap=ori_cmap)
    ax2.imshow(res_img, cmap=res_cmap)

    ax1.set_title(ori_tit, fontsize=fontsize)
    ax2.set_title(res_tit, fontsize=fontsize)


def plot_stack(imgs, titles=None, fontsizes=None, cmaps=None, hv='h', figsize=(15, 9)):
    """
    @hv, plot in a horizontal column or a vertical row 
    """
    if not isinstance(imgs, list): # incase `imgs` in a single image not a list
        imgs = [imgs]
    
    n = len(imgs)
    
    _titles, _fontsizes, _cmaps = ['']*n, [50/n]*n, ['gray']*n
    if titles != None:
        _titles[:len(titles)] = titles
    if fontsizes != None:
        _fontsizes[:len(fontsizes)] = fontsizes 
    if cmaps != None:
        _cmaps[:len(cmaps)] = cmaps 
    
    #print(_titles, _fontsize, _cmaps)
    
    f, axes = plt.subplots(1, n, figsize=figsize)
    f.tight_layout()
    
    # in case `imgs` in a single image not a list
    axes = [axes] if len(imgs) == 1 else axes.tolist()

    for ax, img, cmap, fs, tit in zip(axes, imgs, _cmaps, _fontsizes, _titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(tit, fontsize=fs)

    return f, axes


class PersTrans():
    @staticmethod
    def computeM(pts_src, pts_dst, img=None):
        """
        @pts_src, Coordinates of quadrangle vertices in the source image.
                  Shoud follow the order of up-left, up-right, down-left, down-right for displaying while img != None
        @pts_dst, Coordinates of the corresponding quadrangle vertices in the destination image.
        """

        # Compute perpective transform
        pers_M = cv2.getPerspectiveTransform(pts_src, pts_dst)

        if img is not None:
            # Draw Lines
            cv2.line(img, tuple(pts_src[0]), tuple(
                pts_src[2]), (255, 0, 0), 1, cv2.LINE_AA)  # 左红色直线
            cv2.line(img, tuple(pts_src[1]), tuple(
                pts_src[3]), (255, 0, 0), 1, cv2.LINE_AA)  # 右红色直线

        # Draw pts1, pts2 on original and affined image respectively
        for p1, p2 in zip(pts_src, pts_dst):
            cv2.circle(img, tuple(p1), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p2), 4, (0, 255, 0), -1, cv2.LINE_AA)

        return pers_M

    @staticmethod
    def get_y_from_x(x, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        k = (y2 - y1) / (x2 - x1)
        return k*(x-x2) + y2

    @staticmethod
    def get_x_from_y(y, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        k = (x2 - x1) / (y2 - y1)
        return k*(y-y2) + x2


class CamParams():
    parm_file_prefix = 'CamParams-'

    def __init__(self):
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

        # Compute perpective transform and its inverse
        #self.pers_M = cv2.getPerspectiveTransform(pts1, pts2)
        #self.pers_M_inv = np.linalg.inv(pers_M)

    def get(self):
        return self.mtx, self.dist, self.rvecs, self.tvecs

    def save(self, cam_num):
        import pickle
        fn = CamParams.parm_file_prefix + str(cam_num) + '.pkl'

        # Saving the camera parameters:
        with open(fn, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.mtx,
                         self.dist,
                         self.rvecs,
                         self.tvecs],
                        f)

    def load(self, cam_num):
        import pickle
        fn = CamParams.parm_file_prefix + str(cam_num) + '.pkl'

        # Getting back the objects:
        with open(fn, 'rb') as f:  # Python 3: open(..., 'rb')
            self.mtx, self.dist, self.rvecs, self.tvecs = pickle.load(f)

    def calibration(self, img_list, x, y):
        """
        @img_list, a list of image path to be calibrated
        @x, the number of points along x axes
        @y, the number of points along y axes
        """
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((x*y, 3), np.float32)
        objp[:, :2] = np.mgrid[0:y, 0:x].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for fname in img_list:
            gray = cv2.imread(fname, 0)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                #img = cv2.drawChessboardCorners(gray, (9,6), corners, ret)
                # cv2.imshow('img',img)
                # cv2.waitKey(30)

        # cv2.destroyAllWindows()
        img_shape = cv2.imread(img_list[0], 0).shape[::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None)
        if ret > 1.0:
            print('Warning: retval %f is better to less than 1.0' % ret)

        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

        return ret

    def undistort(self, img_src):
        return cv2.undistort(img_src, self.mtx, self.dist, None, self.mtx)


class EdgeDetector():
    @staticmethod
    def Sobel(img, dx, dy):
        sobel = cv2.Sobel(img, cv2.CV_64F, dx, dy)  # Take the derivative in x
        # Absolute x derivative to accentuate lines away from horizontal
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255.*abs_sobel/np.max(abs_sobel))
        return scaled_sobel

    @staticmethod
    def BinaryThresholding(img, threshold, value=1):
        """
        @threshold, [start, end), note the scale includes `start` but not `end`
        @value, value to be write
        """
        binary = np.zeros_like(img, np.uint8)
        binary[(img >= threshold[0]) & (img < threshold[1])] = value
        #print(binary.shape, binary.dtype)
        return binary


class LaneDetector():
    """
    1. Load the camera calibration matrix and distortion coefficients.
    2. Apply a distortion correction to raw images.
    3. Use color transforms, gradients, etc., to create a thresholded binary image.
    4. Apply a perspective transform to rectify binary image ("birds-eye view").
    5. Detect lane pixels and fit to find the lane boundary.
    6. Determine the curvature of the lane and vehicle position with respect to center.
    7. Warp the detected lane boundaries back onto the original image.
    8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

    Debug:
    1. FrameID, Result Description, Comparison Images
    """

    def __init__(self):
        self.cam_mtx = None
        self.cam_dist = None
        self.pers_M = None
        self.pers_M_inv = None
        self.pers_shape = None
        # self.pers_M_inv2 = None
        # self.pers_shape2 = None
        self.history = [None, None]

    def getLaneHistory(self, idx=-1):
        if idx != -1:
            raise NotImplementedError
        return self.history # [left_fitx, right_fitx]

    def pushLane(self, lane):
        """
        @lane, should be [left_fitx, right_fitx]
        """
        self.history = lane

    def loadCamParams(self, cam_id):
        self.__init__()
        cam_params = CamParams()
        cam_params.load(cam_id)

        self.cam_mtx, self.cam_dist, self.pers_M, self.pers_M_inv = cam_params.get()

        # Compute perpective transform
        pts = [np.float32([[204, 720], [1110, 720], [550, 483], [738, 483]]),
               np.float32([[300, 720], [940, 720], [300, 0], [940, 0]]), 
               (720, 1280)]

        self.pers_M = cv2.getPerspectiveTransform(pts[0], pts[1])
        self.pers_M_inv = np.linalg.inv(self.pers_M)
        self.pers_shape = pts[2]

    @staticmethod
    def KeepROI(img, roi_ltrt_coeff=[0.469, 0.537, 0.510, 0.537]):
        """
        Defining a four sided polygon to mask
        """
        shape_img = img.shape
        img_mask = np.zeros(shape_img, dtype=np.uint8)
        lb = (0, shape_img[0])
        lt = (int(roi_ltrt_coeff[0]*shape_img[1]), int(roi_ltrt_coeff[1]*shape_img[0]))
        rb = (shape_img[1], shape_img[0])
        rt = (int(roi_ltrt_coeff[2]*shape_img[1]), int(roi_ltrt_coeff[3]*shape_img[0]))
        vertices = np.array([[lb, lt, rt, rb]], dtype=np.int32)
        try:
            c = img_rgb.shape[2]
            cv2.fillPoly(img_mask, vertices, [1]*c)
        except:
            cv2.fillPoly(img_mask, vertices, [1])
        
        img[img_mask == 0] = 0

    def preProcess(self, img):
        # Apply undistortion
        img_undist = cv2.undistort(img,
                                   self.cam_mtx,
                                   self.cam_dist,
                                   None,
                                   self.cam_mtx)

        return img_undist

    @staticmethod
    def ComputeConf(left_fit, right_fit, height=720, uniform_sample=10):
        """
        Compute Confidence
        """
        if left_fit == None or right_fit == None:
            return -1000, 0 # conf, similarity
        
        sample_idx = np.array(np.linspace(0, height-1, uniform_sample)).astype(np.int)
        #l = np.array(left_fitx[sample_idx])
        #r = np.array(right_fitx[sample_idx])
        l = left_fit[0]*sample_idx**2 + left_fit[1]*sample_idx + left_fit[2]
        r = right_fit[0]*sample_idx**2 + right_fit[1]*sample_idx + right_fit[2]

        similarity = np.std(l - r)
        center_diff = np.abs(np.mean(l) - np.mean(r)) # 400
        conf = 100 - np.abs(center_diff - 550 + center_diff - 650)/5
        return conf, similarity

    @staticmethod
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

    @staticmethod
    def detEdgePeaks(binary_edges, edge_width=40):
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

        return leftx_base, rightx_base

    @staticmethod
    def ReloadLanePixels(img_edges, left_fit, right_fit, nwindows=9, margin=80, return_result=False):
        """
        @nwindows, Choose the number of sliding windows
        @margin, Set the width of the windows +/- margin
        @return_result, whether to return a image including processing results such as slide windows
        """
        if return_result == False:
            out_img = img_edges
        else:
            out_img = np.dstack((img_edges, img_edges, img_edges))*255
            out_img = np.ascontiguousarray(out_img, dtype=np.uint8)

        if left_fit == None and right_fit == None:
            return [], [], out_img

        ploty = np.linspace(0, img_edges.shape[0]-1, nwindows)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] if left_fit != None else []
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]  if right_fit != None else []

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(img_edges.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img_edges.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        if len(left_fitx) != 0:
            leftx_current = int(left_fitx[-1])
        if len(right_fitx) != 0:
            rightx_current = int(right_fitx[-1])

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img_edges.shape[0] - (window+1)*window_height
            win_y_high = img_edges.shape[0] - window*window_height

            # Find the four below boundaries of the window
            if len(left_fitx) != 0:
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                
                if return_result == True:
                    # Draw the windows on the visualization image
                    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 10)
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                
                leftx_current = int(left_fitx[-(window+1)])
                left_lane_inds.append(good_left_inds)

            if len(right_fitx) != 0:
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                if return_result == True:
                    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 10)
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                rightx_current = int(right_fitx[-(window+1)])
                right_lane_inds.append(good_right_inds)
            
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftxy, rightxy = [], []
        if len(left_lane_inds) > 0:
            leftxy = [nonzerox[left_lane_inds], nonzeroy[left_lane_inds]]
        if len(right_lane_inds) > 0:
            rightxy = [nonzerox[right_lane_inds], nonzeroy[right_lane_inds]]

        return leftxy, rightxy, out_img

    @staticmethod
    def ComputePolynomial(var_max, coef_l, coef_r):
        # Generate x and y values for plotting
        ploty = np.linspace(0, base_axis-1, binary_warped.shape[0])
        
        left_fitx = coef_l[0]*ploty**2 + coef_l[1]*ploty + coef_l[2] if coef_l != None else []
        right_fitx = coef_r[0]*ploty**2 + coef_r[1]*ploty + coef_r[2]  if coef_r != None else []

        return left_fitx, right_fitx

    @staticmethod
    def FitPolynomial(left_xy, right_xy):
        if len(left_xy) != 0:
            # Fit a second order polynomial to each using `np.polyfit`
            left_fit = np.polyfit(left_xy[1], left_xy[0], 2).tolist()
        else:
            left_fit = None

        if len(right_xy) != 0:
            right_fit = np.polyfit(right_xy[1], right_xy[0], 2).tolist()
        else:
            right_fit = None

        return left_fit, right_fit

    @staticmethod
    def DrawPolynomialLanes(inout_img, lx, rx):
        ploty = np.linspace(0, inout_img.shape[0]-1, inout_img.shape[0])

        for xl, xr, y in zip(lx.astype(np.int), rx.astype(np.int), ploty.astype(np.int)):
            cv2.circle(inout_img, (xl, y), 3, (0, 0, 255), -1)
            cv2.circle(inout_img, (xr, y), 3, (0, 0, 255), -1)

        return inout_img

    @staticmethod
    def FindLanePixels(img_edges, leftx_base, rightx_base, nwindows=9, margin=80, minpix=10, return_result=False):
        """
        @nwindows, Choose the number of sliding windows
        @margin, Set the width of the windows +/- margin
        @minpix, Set minimum number of pixels found to recenter window
        @return_result, whether to return a image including processing results such as slide windows
        """
        if return_result == False:
            out_img = img_edges
        else:
            out_img = np.dstack((img_edges, img_edges, img_edges))*255
            out_img = np.ascontiguousarray(out_img, dtype=np.uint8)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img_edges, img_edges, img_edges))*255
        out_img = np.ascontiguousarray(out_img, dtype=np.uint8)

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(img_edges.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img_edges.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        window = 0; iter_again = -1; force_l, force_r = 0, 0; _force_l, _force_r = 0, 0
        while window < nwindows:
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img_edges.shape[0] - (window+1)*window_height
            win_y_high = img_edges.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            if return_result == True:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 10)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 10)

            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # If you found > minpix pixels, recenter next window 
            # (`right` or `leftx_current`) on their mean position
            if(good_left_inds.size > minpix) and (iter_again != 0):
                _force_l = np.mean(nonzerox[good_left_inds]) - leftx_current # current
                tmp_ofs = (force_l + _force_l) / 2.
                leftx_current += np.int(tmp_ofs)
                iter_again = 1
            elif iter_again == 0:
                if window < 3: # 一开始禁止使用动量
                    force_r = 0
                else:
                    force_l = _force_l*0.8 if good_left_inds.size > minpix else _force_l/3.
                
            if(good_right_inds.size > minpix) and (iter_again != 0):
                _force_r = np.mean(nonzerox[good_right_inds]) - rightx_current # current
                tmp_ofs = (force_r + _force_r) / 2.
                rightx_current += np.int(tmp_ofs)
                iter_again = 1
            elif iter_again == 0:
                if window < 3: # 一开始禁止使用动量
                    force_r = 0
                else:
                    force_r = _force_r*0.8 if good_right_inds.size > minpix else _force_r/3.
                    
            if iter_again == 1:
                iter_again = 0
            else:
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                leftx_current += np.int(force_l)
                rightx_current += np.int(force_r)
                iter_again = -1
                window += 1                


        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftxy, rightxy = [], []
        if len(left_lane_inds) > 0:
            leftxy = [nonzerox[left_lane_inds], nonzeroy[left_lane_inds]]
        if len(right_lane_inds) > 0:
            rightxy = [nonzerox[right_lane_inds], nonzeroy[right_lane_inds]]

        return leftxy, rightxy, out_img