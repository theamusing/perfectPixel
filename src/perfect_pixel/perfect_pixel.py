import numpy as np
import cv2

def compute_fft_magnitude(gray_image):
    f = np.fft.fft2(gray_image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag = 1 - np.log1p(mag)  # log(1 + |F|)
    # normalize to [0, 1]
    mn, mx = float(mag.min()), float(mag.max())
    if mx - mn < 1e-8:
        return np.zeros_like(mag, dtype=np.float32)
    mag = (mag - mn) / (mx - mn)
    return mag

def smooth_1d(v, k = 17):
    """Simple 1D smoothing with a Gaussian-like kernel (no scipy)."""
    k = int(k)
    if k < 3:
        return v
    if k % 2 == 0:
        k += 1
    sigma = k / 6.0
    x = np.arange(k) - k // 2
    ker = np.exp(-(x * x) / (2 * sigma * sigma))
    ker = ker / (ker.sum() + 1e-8)
    vv = np.convolve(v, ker, mode="same")
    return vv

def detect_peak(proj, peak_width = 6, rel_thr=0.35, min_dist=6):
    center = len(proj) // 2

    mx = float(proj.max())
    if mx < 1e-6:
        return None

    thr = mx * float(rel_thr)
    
    candidates = []
    for i in range(1, len(proj) - 1):
        is_peak = True
        for j in range(1, peak_width):
            if i - j < 0 or i + j >= len(proj):
                continue
            if proj[i-j+1] < proj[i - j] or proj[i+j-1] < proj[i + j]:
                is_peak = False
                break  
        if is_peak and proj[i] >= thr:
            left_climb = 0
            for k in range(i, 0, -1):
                if proj[k] > proj[k-1]:
                    left_climb = abs(proj[i] - proj[k-1])
                else:
                    break

            right_fall = 0
            for k in range(i, len(proj) - 1):
                if proj[k] > proj[k+1]:
                    right_fall = abs(proj[i] - proj[k+1])
                else:
                    break
            
            candidates.append({
                'index': i,
                'climb': left_climb,
                'fall': right_fall,
                'score': max(left_climb, right_fall)
            })

    if not candidates:
        return None

    # enforce a dead-zone around center
    left = [i for i in candidates if i['index'] < center - min_dist and i['index'] > center * 0.25]
    right = [i for i in candidates if i['index'] > center + min_dist and i['index'] < center * 1.75]

    left.sort(key=lambda x: x['score'], reverse=True)
    right.sort(key=lambda x: x['score'], reverse=True)

    if not left or not right:
        return None

    # pick nearest to center on each side
    peak_left = left[0]['index']   
    peak_right = right[0]['index']  

    return abs(peak_right - peak_left)/2

def find_best_grid(origin, range_val_min, range_val_max, grad_mag, thr = 0):
    best = round(origin)
    peaks = []
    mx = np.max(grad_mag)
    if mx < 1e-6:
        return best
    rel_thr = mx * thr
    for i in range(-round(range_val_min), round(range_val_max)+1):
        candidate = round(origin + i)
        if candidate <= 0 or candidate >= len(grad_mag) - 1:
            continue
        if grad_mag[candidate] > grad_mag[candidate -1] and grad_mag[candidate] > grad_mag[candidate +1] and grad_mag[candidate] >= rel_thr:
            peaks.append((grad_mag[candidate], candidate))
    if len(peaks) == 0:
        return best
    
    # find the brightest peak
    peaks.sort(key=lambda x: x[0], reverse=True)
    best = peaks[0][1]
    return best

def sample_center(image, x_coords, y_coords):
    x = np.asarray(x_coords)
    y = np.asarray(y_coords)

    centers_x = np.clip((x[1:] + x[:-1]) * 0.5, 0, image.shape[1] - 1).astype(np.int32)
    centers_y = np.clip((y[1:] + y[:-1]) * 0.5, 0, image.shape[0] - 1).astype(np.int32)

    scaled_image = image[centers_y[:, None], centers_x[None, :]]
    return scaled_image

def sample_majority(image, x_coords, y_coords, max_samples=128, iters=6, seed=42):
    rng = np.random.default_rng(seed)

    img = image.astype(np.float32) if image.dtype != np.float32 else image
    H, W = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]
    C = img.shape[2]

    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)

    nx, ny = len(x) - 1, len(y) - 1
    out = np.empty((ny, nx, C), dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1.0)
    
    for j in range(ny):
        y0, y1 = int(y[j]), int(y[j + 1])
        y0 = np.clip(y0, 0, H); y1 = np.clip(y1, 0, H)
        if y1 <= y0: y1 = min(y0 + 1, H)

        for i in range(nx):
            x0, x1 = int(x[i]), int(x[i + 1])
            x0 = np.clip(x0, 0, W); x1 = np.clip(x1, 0, W)
            if x1 <= x0: x1 = min(x0 + 1, W)

            cell = img[y0:y1, x0:x1].reshape(-1, C)
            n = cell.shape[0]
            if n == 0:
                out[j, i] = 0
                continue
            if n > max_samples:
                cell = cell[rng.integers(0, n, size=max_samples)]

            if cell.shape[0] < 2:
                out[j, i] = cell[0]
            else:
                # 使用 cv2.kmeans 聚成 2 类
                # K=2, attempts=1, 使用 KMEANS_RANDOM_CENTERS 或 PP 模式
                _, labels, centers = cv2.kmeans(
                    cell, 2, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS
                )
                
                # 计算两个簇的像素数量，labels 是二维数组 (N, 1)
                count1 = np.sum(labels)  # 标签是 0 和 1
                count0 = len(labels) - count1
                
                # 多数表决：取成员较多的中心点
                out[j, i] = centers[1] if count1 >= count0 else centers[0]
            # --- 替换部分结束 ---

    if image.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out

def sample_median(image, x_coords, y_coords):
    img = image.astype(np.float32) if image.dtype != np.float32 else image
    H, W = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]
    C = img.shape[2]

    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)

    nx, ny = len(x) - 1, len(y) - 1
    out = np.empty((ny, nx, C), dtype=np.float32)
    
    for j in range(ny):
        y0, y1 = int(y[j]), int(y[j + 1])
        y0 = np.clip(y0, 0, H); y1 = np.clip(y1, 0, H)
        if y1 <= y0: y1 = min(y0 + 1, H)

        for i in range(nx):
            x0, x1 = int(x[i]), int(x[i + 1])
            x0 = np.clip(x0, 0, W); x1 = np.clip(x1, 0, W)
            if x1 <= x0: x1 = min(x0 + 1, W)

            cell = img[y0:y1, x0:x1].reshape(-1, C)
            if cell.shape[0] == 0:
                out[j, i] = 0
            else:
                out[j, i] = np.median(cell, axis=0)

    if image.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out

def refine_grids(image, grid_x, grid_y, refine_intensity=0.25):
    H, W = image.shape[:2]
    x_coords = []
    y_coords = []
    cell_w = W / grid_x
    cell_h = H / grid_y

    # calculate gradient magnitude
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    grad_x_sum = np.sum(np.abs(grad_x), axis=0).reshape(-1)
    grad_y_sum = np.sum(np.abs(grad_y), axis=1).reshape(-1)

    # refine grid lines based on gradient magnitude from center
    x = find_best_grid(W / 2, cell_w, cell_w, grad_x_sum)
    while(x < W + cell_w/2):
        x = find_best_grid(x, cell_w * refine_intensity, cell_w * refine_intensity, grad_x_sum)
        x_coords.append(x)
        x += cell_w
    x = find_best_grid(W / 2, cell_w, cell_w, grad_x_sum) - cell_w
    while(x > -cell_w/2):
        x = find_best_grid(x, cell_w * refine_intensity, cell_w * refine_intensity, grad_x_sum)
        x_coords.append(x)
        x -= cell_w

    y = find_best_grid(H / 2, cell_h, cell_h, grad_y_sum)
    while(y < H + cell_h/2):
        y = find_best_grid(y, cell_h * refine_intensity, cell_h * refine_intensity, grad_y_sum)   
        y_coords.append(y)
        y += cell_h
    y = find_best_grid(H / 2, cell_h, cell_h, grad_y_sum) - cell_h
    while(y > -cell_h/2):
        y = find_best_grid(y, cell_h * refine_intensity, cell_h * refine_intensity, grad_y_sum)   
        y_coords.append(y)
        y -= cell_h
    
    x_coords = sorted(x_coords)
    y_coords = sorted(y_coords)

    return x_coords, y_coords

def estimate_grid_fft(gray, peak_width=6):
    """Return (grid_w, grid_h) or None."""
    H, W = gray.shape

    mag = compute_fft_magnitude(gray)

    band_row = W // 2
    band_col = H // 2
    row_sum = np.sum(mag[:, W//2 - band_row: W//2 + band_row], axis=1)
    col_sum = np.sum(mag[H//2 - band_col: H//2 + band_col, :], axis=0)

    row_sum = cv2.normalize(row_sum, None, 0, 1, cv2.NORM_MINMAX).flatten()
    col_sum = cv2.normalize(col_sum, None, 0, 1, cv2.NORM_MINMAX).flatten()

    row_sum = smooth_1d(row_sum, k=17)
    col_sum = smooth_1d(col_sum, k=17)

    scale_row = detect_peak(row_sum, peak_width)
    scale_col = detect_peak(col_sum, peak_width)

    if (
        scale_row is None
        or scale_col is None
        or scale_row <= 0
        or scale_col <= 0
    ):
        return None, None

    grid_w = int(round(scale_col))
    grid_h = int(round(scale_row))
    return grid_w, grid_h

def estimate_grid_gradient(gray, rel_thr=0.2):
    H, W = gray.shape

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    grad_x_sum = np.sum(np.abs(grad_x), axis=0).reshape(-1)
    grad_y_sum = np.sum(np.abs(grad_y), axis=1).reshape(-1)

    peak_x = []
    peak_y = []

    thr_x = float(rel_thr) * float(grad_x_sum.max())
    thr_y = float(rel_thr) * float(grad_y_sum.max())

    min_interval = 4
    for i in range(1, len(grad_x_sum) - 1):
        if grad_x_sum[i] > grad_x_sum[i - 1] and grad_x_sum[i] > grad_x_sum[i + 1] and grad_x_sum[i] >= thr_x:
            if len(peak_x) == 0 or i - peak_x[-1] >= min_interval:
                peak_x.append(i)

    for i in range(1, len(grad_y_sum) - 1):
        if grad_y_sum[i] > grad_y_sum[i - 1] and grad_y_sum[i] > grad_y_sum[i + 1] and grad_y_sum[i] >= thr_y:
            if len(peak_y) == 0 or i - peak_y[-1] >= min_interval:
                peak_y.append(i)

    if len(peak_x) < 4 or len(peak_y) < 4:
        return None, None

    # get median interval
    intervals_x = []
    for i in range(1, len(peak_x)):
        intervals_x.append(peak_x[i] - peak_x[i - 1])
    intervals_y = []
    for i in range(1, len(peak_y)):
        intervals_y.append(peak_y[i] - peak_y[i - 1])
    
    scale_x = W / np.median(intervals_x)
    scale_y = H / np.median(intervals_y)

    print(f"Detected grid size from gradient: ({scale_x:.2f}, {scale_y:.2f})")

    return int(round(scale_x)), int(round(scale_y))

def detect_grid_scale(image, peak_width=6, max_ratio=1.5, min_size=4.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    grid_w, grid_h =  estimate_grid_fft(gray, peak_width=peak_width)
    if grid_w is None or grid_h is None:
        print("FFT-based grid estimation failed, fallback to gradient-based method.")
        grid_w, grid_h = estimate_grid_gradient(gray)
    else:
        pixel_size_x = W / grid_w
        pixel_size_y = H / grid_h
        max_pixel_size = 20.0
        if min(pixel_size_x, pixel_size_y) < min_size or max(pixel_size_x, pixel_size_y) > max_pixel_size or pixel_size_x / pixel_size_y > max_ratio or pixel_size_y / pixel_size_x > max_ratio:
            print("Inconsistent grid size detected (FFT-based), fallback to gradient-based method.")
            grid_w, grid_h = estimate_grid_gradient(gray)

    if grid_w is None or grid_h is None:
        print("Gradient-based grid estimation failed.")
        return None, None
    
    pixel_size_x = W / grid_w
    pixel_size_y = H / grid_h

    if pixel_size_x / pixel_size_y > max_ratio or pixel_size_y / pixel_size_x > max_ratio:
        pixel_size = min(pixel_size_x, pixel_size_y)
    else:   
        pixel_size = (pixel_size_x + pixel_size_y) / 2.0

    print(f"Detected pixel size: {pixel_size:.2f}")

    grid_w = int(round(W / pixel_size))
    grid_h = int(round(H / pixel_size))

    return grid_w, grid_h

def grid_layout(image, x_coords, y_coords, scale_x, scale_y):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(image)
    plt.title(f"Scaled Image by Grid Sampling({scale_x}x{scale_y})")
    for x in x_coords:
        plt.axvline(x=x, linewidth=0.6)
    for y in y_coords:
        plt.axhline(y=y, linewidth=0.6)
    plt.show()

def get_perfect_pixel(image, sample_method="center", grid_size = None, min_size = 4.0, peak_width = 6, refine_intensity = 0.25, fix_square = True, preprocess_denoise=True, bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75, debug=False):
    """
    Args:
        image: RGB Image (H * W * 3)
        sample_method: "majority", "center", or "median"
        grid_size: Manually set grid size (grid_w, grid_h) to override auto-detection
        min_size: Minimum pixel size to consider valid
        peak_width: Minimum peak width for peak detection.
        refine_intensity: Intensity for grid line refinement. Recommended range is [0, 0.5]. Given original estimated grid line at x, the refinement will search in [x * (1 - refine_intensity), x * (1 + refine_intensity)].
        fix_square: Whether to enforce output to be square when detected image is almost square.
        preprocess_denoise: Whether to apply a bilateral filter before sampling. Greatly helps with noise. Note: The filtered image is used for both detection and sampling, but the original input image array passed to this function is not modified in-place.
        bilateral_d: Diameter of each pixel neighborhood that is used during filtering.
        bilateral_sigma_color: Filter sigma in the color space.
        bilateral_sigma_space: Filter sigma in the coordinate space.
        debug: Whether to show debug plots.

    returns: 
        refined_w, refined_h, scaled_image
    """
    if preprocess_denoise:
        image = cv2.bilateralFilter(image, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

    H, W = image.shape[:2]
    if grid_size is not None:
        # use provided grid size
        scale_col, scale_row = grid_size
    else:
        scale_col, scale_row = detect_grid_scale(image, peak_width=peak_width, max_ratio=1.5, min_size=min_size)
        if scale_col is None or scale_row is None:
            print("Failed to estimate grid size.")
            return None, None, image

    size_x = int(round(scale_col))
    size_y = int(round(scale_row))
    x_coords, y_coords = refine_grids(image, size_x, size_y, refine_intensity)

    refined_size_x = len(x_coords) - 1
    refined_size_y = len(y_coords) - 1

    # sample
    if sample_method == "majority":
        scaled_image = sample_majority(image, x_coords, y_coords)
    elif sample_method == "median":
        scaled_image = sample_median(image, x_coords, y_coords)
    else:
        # sample by center
        scaled_image = sample_center(image, x_coords, y_coords)

    # fix square
    if fix_square and abs(refined_size_x - refined_size_y) == 1:
        # align to even sized square
        if refined_size_x > refined_size_y:
            if refined_size_x % 2 == 1:
                # remove one column
                scaled_image = scaled_image[:, :-1]
            else:
                # add one row by duplicating first row
                scaled_image = np.concatenate([scaled_image[:1, :], scaled_image], axis=0)
        else:
            if refined_size_y % 2 == 1:
                # remove one row
                scaled_image = scaled_image[:-1, :]
            else:
                # add one col by duplicating first col
                scaled_image = np.concatenate([scaled_image[:, :1], scaled_image], axis=1)
    refined_size_y, refined_size_x = scaled_image.shape[:2]
    print(f"Refined grid size: ({refined_size_x}, {refined_size_y})")

    # debug
    if debug:
        grid_layout(image, x_coords, y_coords, refined_size_x, refined_size_y)

    return refined_size_x, refined_size_y, scaled_image
