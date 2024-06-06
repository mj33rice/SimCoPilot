from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Task 1:
#resize two imgs
#resize imgA with cv2.resize
#resize imgB via center cropping
#ensure both resized images are resized to same end_size
#concatenate the resized imgs via every other row imgA and every other row imgB

A = Image.open('./imgA.jpg')
B = Image.open('./imgB.jpg')
A_array = np.array(A)
B_array = np.array(B)
print('Array shapes:',A_array.shape, B_array.shape)
print(A_array)
print(B_array)

end_size = 256

if end_size >= A_array.shape[0] or end_size>= B_array.shape[0]:
    print('choose end size less than: ', np.min(A_array.shape,B_array.shape))

A_resized = cv2.resize(A_array, (end_size, end_size))

def center_crop(img_array, end_size):
    x_start = int((img_array.shape[0]-end_size)/2)
    x_end = x_start + end_size

    y_start = int((img_array.shape[1]-end_size)/2)
    y_end = y_start + end_size

    img_resized = img_array[x_start:x_end, y_start:y_end, :]
    return img_resized
B_resized = center_crop(B_array, end_size)
print(B_resized.shape)
C = np.concatenate((A_resized[0:256,0:128,:],B_resized[0:256,128:256,:]),axis = 1)

D = B_resized[0:1,:,:]
for row in range(1,A_resized.shape[0]):
    if row % 2 == 0:
        D = np.concatenate((D,B_resized[row:row+1,:,:]), axis=0)
    else:
        D = np.concatenate((D,A_resized[row:row+1,:,:]), axis =0)
print(D)

#Task 2:
#upload picture of multiple peppers each different colors
#create a mask without the yellow peppers by using range provided
#do this in rgb and hsv -> note different ranges depending on rgb or hsv
pepper_img = Image.open('./pepper.png')
pepper = np.array(pepper_img)

lower_yellow = np.array([150, 175, 0], dtype=np.uint8)
upper_yellow = np.array([255, 255, 150], dtype=np.uint8)
mask = np.all((pepper >= lower_yellow) & (pepper <= upper_yellow), axis=-1)

result = np.where(mask, 1, 0)

print(pepper)
print(result)

img = cv2.imread('./pepper.png')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([19, 0, 0], dtype=np.uint8)
upper_yellow = np.array([24, 255, 255], dtype=np.uint8)
mask = np.all((hsv_img >= lower_yellow) & (hsv_img <= upper_yellow), axis=-1)

result_hsv = np.where(mask, 1, 0)

print(hsv_img)
print(result_hsv)

#Task 3:
#write transormation functions to translsate, rotate, and perform similarity and affine transformation
#write bilinear interpolation function from scratch
#apply series of transformations to an image

def translation(dx,dy):
    translation_matrix = np.array([[1,0,dx],[0,1,dy],[0,0,1]])
    return translation_matrix

def rotation(angle,radians = True):
    if radians == False:
        angle = np.radians(angle)
    costheta = np.cos(angle)
    sintheta = np.sin(angle)
    rotation_matrix = np.array([[costheta, sintheta,0],[-1*sintheta,costheta,0],[0,0,1]])
    return rotation_matrix

def similarity_matrix(angle, dx, dy, scale_factor,radians=True):
    if radians == False:
        angle = np.radians(angle)
    costheta = np.cos(angle)
    sintheta = np.sin(angle)

    similarity_matrix = np.array([[scale_factor*costheta,scale_factor*sintheta,dx],
                                [-1*scale_factor*sintheta, scale_factor*costheta, dy],
                                [0,0,1]])
    return similarity_matrix

def affine(angle, x, y, scale, ax, ay):
    scaling = np.array([[scale, 0,0], [0, scale, 0], [0,0,1]])
    shear = np.array([[1, ax, 0], [ay, 1,0], [0, 0,1]])
    result = np.array([[0,0,0], [0,0,0], [0,0,0]])
    result = np.dot(translation(x, y), rotation(angle))
    result = np.dot(result, scaling)
    result = np.dot(result, shear)
    return result

def bilinear_interpolation(image,x,y):
    x1 = int(x)
    x2 = x1 + 1
    y1 = int(y)
    y2 = y1 + 1

    if x1 < 0 or y1 < 0 or x2 >= image.shape[1] or y2 >= image.shape[0]:
        return 0
    else:
        f11 = image[y1][x1]
        f12 = image[y1][x2]
        f21 = image[y2][x1]
        f22 = image[y2][x2]

        w1 = (x2-x)*(y2-y)
        w2 = (x-x1)*(y2-y)
        w3 = (x2-x)*(y-y1)
        w4 = (x-x1)*(y-y1)

    return (w1*f11) + (w2*f12) + (w3*f21) + (w4*f22)

def image_warp(I,T):
    rows,cols = I.shape[:2]
    output = np.zeros((rows,cols,3))
    center = (cols/2, rows/2)
    T_invert = np.linalg.inv(T)

    for i in range(rows):
        for j in range(cols):
            shift_center = np.array([j-center[0],i -center[1],1])
            coordinates = np.dot(T_invert,shift_center)
            x,y = coordinates[0] + center[0], coordinates [1] + center[1]
            output[i][j] = bilinear_interpolation(I,x,y)
    output = np.array(output, np.uint8)
    return output

path= './arabella.jpg'
arabella = cv2.imread(path)
arabella_smol = cv2.resize(arabella, dsize=(256, 192), interpolation=cv2.INTER_AREA)
arabella_smol = np.array(arabella_smol)
arabella_smol = arabella_smol[:, :, [2, 1, 0]]

#translate images keep params as shown
t1 = translation(21,25)
warped_arabella1 = image_warp(arabella_smol,t1)
t2 = translation(-21,25)
warped_arabella2 = image_warp(arabella_smol,t2)
t3 = translation(21,-25)
warped_arabella3 = image_warp(arabella_smol,t3)
t4 = translation(-21,25)
warped_arabella4 = image_warp(arabella_smol,t4)
print(warped_arabella1)
print(warped_arabella2)
print(warped_arabella3)
print(warped_arabella4)

# rotate image 30 degrees clockwise and 30 degrees counterclockwise
r1 = rotation(30, False)
r2 = rotation(-30, False)

warped_arabella5 = image_warp(arabella_smol,r1)
warped_arabella6 = image_warp(arabella_smol,r2)
print(warped_arabella5)
print(warped_arabella6)

#apply similarity transformation to image, keep params as shown below
s1 = similarity_matrix(60, 0, 0, 0.5,radians=False)
warped_arabella7 = image_warp(arabella_smol,s1)
print(warped_arabella7)

#apply affine transformation to image, keep params as shown below
a1 = affine(90, 2, 3, .5, 5, 2)
warped_arabella8 = image_warp(arabella_smol,a1)
print(warped_arabella8)

#Task 4:
#artificially replicate overhead and desklight scene via addition of overhead lit only and desklight lit only scenes
#make sure to scale properly 
path1= './desklight.jpg'
path2= './overheadlight.jpg'
path3 = './bothlight.jpg'
I1 = np.array(cv2.imread(path1))[:,:,[2,1,0]]
I2 = np.array(cv2.imread(path2))[:,:,[2,1,0]]
I12 = np.array(cv2.imread(path3))[:,:,[2,1,0]]

type(I12[0,0,0])
I1_float = I1/255.0
I2_float = I2/255.0
I12_float = I1_float + I2_float
type(I12_float[0,0,0]),np.min(I12_float),np.max(I12_float)
I12_uint8 = (I12_float * 255.0).astype(np.uint8)
type(I12_uint8[0,0,0]),np.min(I12_uint8),np.max(I12_uint8)

synthI12 = I1+I2
diffI = synthI12 - I12
diffI_scaled = (diffI - np.min(diffI))/(np.max(diffI)-np.min(diffI))

print(I12)
print(synthI12)
print(diffI_scaled)