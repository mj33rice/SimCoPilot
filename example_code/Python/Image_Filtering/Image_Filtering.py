import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.signal import butter, filtfilt

# Task1
# Gaussian blurring einstein monroe illusion
def gaussian2D(sigma, kernel_size):
    sigma_x, sigma_y = sigma
    size_x, size_y = kernel_size

    size_x = int(size_x) // 2 * 2 + 1
    size_y = int(size_y) // 2 * 2 + 1

    x, y = np.meshgrid(np.linspace(-3*sigma_x, 3*sigma_x, size_x),
                        np.linspace(-3*sigma_y, 3*sigma_y, size_y))
    
    kernel = np.exp(-(x**2 / (2*sigma_x**2) + y**2 / (2*sigma_y**2)))
    kernel /= 2 * np.pi * sigma_x * sigma_y
    kernel /= kernel.sum()
    return kernel

def center_crop(image, target_size):
    image = np.array(image)
    h, w = image.shape[:2]

    left = (w - target_size[0]) // 2
    top = (h - target_size[1]) // 2

    right = left + target_size[0]
    bottom = top + target_size[1]

    cropped_image = image[top:bottom, 0:right]
    return cropped_image

img_a = cv2.imread('./marilyn.jpeg')
img_b = cv2.imread('./einstein.jpeg')

# Part1
# Reshape to ensure images are same size by center cropping to image with smallest dimension
# Convert img to grayscale 
smallest_dim = min(img_a.shape[0],img_a.shape[1],img_b.shape[1],img_b.shape[1])
img_a_gray = center_crop(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY),(smallest_dim, smallest_dim))
img_b_gray = center_crop(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY),(smallest_dim, smallest_dim))
print(np.array(img_a_gray))
print(np.array(img_b_gray))

# Part2
# Apply Gaussian filter to both images choose relevant sigma and kernel size to achieve desired results
# Use custom gaussian2D function and use cv2.filter2D to apply the filter to the image
sigma_a = (1, 1)
kernel_size_a = (11, 11)
gaussian_kernel_a = gaussian2D(sigma_a, kernel_size_a)

sigma_b = (1, 1)
kernel_size_b = (11, 11)
gaussian_kernel_b = gaussian2D(sigma_b, kernel_size_b)

blur_a = cv2.filter2D(img_a_gray, -1, gaussian_kernel_a)
blur_b = cv2.filter2D(img_b_gray, -1, gaussian_kernel_b)

a_diff = img_a_gray - blur_a
img_c = blur_b + a_diff
print(img_c)

def downsample_image(image, factor):
    if factor <= 0:
        raise ValueError("Downsampling factor must be greater than 0.")
    
    height, width = image.shape[:2]
    new_height = height // factor
    new_width = width // factor

    if len(image.shape) == 3:
        downsampled_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
    else:
        downsampled_image = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            downsampled_image[i, j] = image[i * factor, j * factor]
    return downsampled_image

downsampling_factor = 4
downsampled_image = downsample_image(img_c, downsampling_factor)
print(np.array(downsampled_image))

# Part3
# Computer fourier magnitude for the final image, original grayscale images, 
# the blurred second image and difference between grayscale first image and blurred first image

def compute_fourier_magnitude(image):
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    log_spectrum = np.log(1 + spectrum)
    return log_spectrum

spectrum_A = compute_fourier_magnitude(img_a_gray)
spectrum_B = compute_fourier_magnitude(img_b_gray)
spectrum_blurred_B = compute_fourier_magnitude(blur_b)
spectrum_A_blur_A = compute_fourier_magnitude(a_diff)
spectrum_C = compute_fourier_magnitude(img_c)
print(spectrum_A)
print(spectrum_B)
print(spectrum_A_blur_A)
print(spectrum_blurred_B)
print(spectrum_C)

# Blend two images with a verticle blend in the middle with laplacian pyramids
# Part 1 vertical blending halfway through image
apple = cv2.imread('./apple.jpeg')
orange = cv2.imread('./orange.jpeg')
A = cv2.resize(apple, (256,256), fx=0.5, fy=0.5)
B = cv2.resize(orange, (256,256), fx=0.5, fy=0.5)

G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# Laplacian Pyramid for A and B
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# Add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
    LS.append(ls)

# Reconstruct
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# Image with direct connecting each half
real = np.hstack((A[:,:cols//2],B[:,cols//2:]))

blended_rgb = cv2.cvtColor(ls_, cv2.COLOR_BGR2RGB)
original_rgb = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

print(blended_rgb)
print(original_rgb)

# Part 2
# Blend the image diagonally in a strip following the same steps as above 
# to accomplish diagnoal blending, use a diagonal mask 
# Create diagonal mask
def create_diagonal_mask(shape, strip_width=200):
    mask = np.zeros(shape, dtype=np.float32)
    height, width, _ = mask.shape
    for i in range(min(height, width)):
        mask[i, max(0, i - strip_width // 2):min(width, i + strip_width // 2), :] = 1.0
    return mask

# Now blend images using the diagonal mask
LS = []
mask = create_diagonal_mask(A.shape)
M = mask.copy()
gpmask = [M]
for i in range(5):
    M = cv2.pyrDown(M)
    gpmask.append(M)
gpmask.reverse()
for i in range(len(gpmask)):
    rows, cols, dpt = lpA[i].shape
    ls = lpA[i] * gpmask[i] + lpB[i] * (1 - gpmask[i])
    LS.append(ls)

# Now reconstruct
ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.resize(ls_, (LS[i].shape[1], LS[i].shape[0])) 
    ls_ = cv2.add(ls_, LS[i])

# Image with direct connecting each diagonal half
real = np.hstack((A[:, :cols//2], B[:, cols//2:]))
ls_rgb = cv2.cvtColor(ls_.astype(np.uint8), cv2.COLOR_BGR2RGB)
real_rgb = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

print(ls_rgb)
print(real_rgb)
print(mask)


# Task3
# Part1
# Read in a video file in .avi format, choose areas of the face to focus on via bounding box
# Apply a bandpass filter to the specified regions of the interest based on a lower and upper bound
def read_video_into_numpy(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
    # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    # Converts to numpy array(T,H,W,C)
    video = np.stack(frames, axis=0)
    # (T,H,W,C)->(H,W,C,T)
    video = np.transpose(video, (1,2,3,0))
    return frames

def bandpass_filter(signal, low_cutoff, high_cutoff, fs, order):
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

alice = './alice.avi'
video_frames = read_video_into_numpy(alice)
first_frame = video_frames[0]

# Specify regions of interest
cheek_rect = [(220, 250), (320, 350)]
forehead_rect = [(220, 10), (500, 174)]
cheek_roi = first_frame[cheek_rect[0][1]:cheek_rect[1][1], cheek_rect[0][0]:cheek_rect[1][0]]
forehead_roi = first_frame[forehead_rect[0][1]:forehead_rect[1][1], forehead_rect[0][0]:forehead_rect[1][0]]

print(cheek_roi)
print(forehead_roi)

# Part 2
# Find the average green value for each frame in the cheek and forhead region of interest
cheek_avg_green_values = []
forehead_avg_green_values = []

for frame in video_frames:
    cheek_roi = frame[cheek_rect[0][1]:cheek_rect[1][1], cheek_rect[0][0]:cheek_rect[1][0]]
    forehead_roi = frame[forehead_rect[0][1]:forehead_rect[1][1], forehead_rect[0][0]:forehead_rect[1][0]]
    cheek_avg_green = np.mean(cheek_roi[:, :, 1])
    forehead_avg_green = np.mean(forehead_roi[:, :, 1])
    cheek_avg_green_values.append(cheek_avg_green)
    forehead_avg_green_values.append(forehead_avg_green)

print(cheek_avg_green_values)
print(forehead_avg_green_values)

# Part3
# Set a lower and upper threshold and apply a bandpass filter to the average green values of cheek and forward 
# Set fs to 30

low_cutoff = 0.8
high_cutoff = 3
fs = 30
order = 1

cheek_filtered_signal = bandpass_filter(cheek_avg_green_values, low_cutoff, high_cutoff, fs, order)
forehead_filtered_signal = bandpass_filter(forehead_avg_green_values, low_cutoff, high_cutoff, fs, order)

print(cheek_filtered_signal)
print(forehead_filtered_signal)

# Part4
# Plot the Fourier magnitudes of these two signals using the DFT, where the x-axis is
# frequency (in Hertz) and y-axis is amplitude. DFT coefficients are ordered in terms of
# integer indices, so you will have to convert the indices into Hertz. For each index n = [-
# N/2, N/2], the corresponding frequency is Fs * n / N, where N is the length of your signal
# and Fs is the sampling rate of the signal (30 Hz in this case). You can also use
# numpy.fft.fftfreq to do this conversion for you.

cheek_fft = np.fft.fft(cheek_filtered_signal)
forehead_fft = np.fft.fft(forehead_filtered_signal)
print(cheek_fft)
print(forehead_fft)

N = len(cheek_filtered_signal)
Fs = 30
freq_cheek = np.fft.fftfreq(N, d=1/Fs)
freq_forehead = np.fft.fftfreq(N, d=1/Fs)
print(np.abs(freq_cheek))
print(np.abs(freq_forehead))

# Part5
# Estimate the pulse rate by finding the index where np.abs(cheek_fft) is at it's maximum
# Cheek heart rate will be aprox 60*freq_cheek[index of max np.abs(cheek_fft)] -> same idea with forhead

index_max_cheek = np.argmax(np.abs(cheek_fft))
index_max_forehead = np.argmax(np.abs(forehead_fft))

freq_max_cheek = freq_cheek[index_max_cheek]
freq_max_forehead = freq_forehead[index_max_forehead]

heart_rate_cheek = (freq_max_cheek) * 60
heart_rate_forehead = (freq_max_forehead) * 60

print(f"Heart Rate (Cheek): {heart_rate_cheek:.2f} beats per minute")
print(f"Heart Rate (Forehead): {heart_rate_forehead:.2f} beats per minute")