import numpy as np
from numpy.polynomial import polynomial as P
from PIL import Image, ImageDraw
import math
import sys
from multiprocessing import Pool
import os
import datetime
#import time
#start = time.time()

WIDTH = int(1080/2)
HEIGHT = int(1080/2)

#self.colors = [ [32,44,57], [200,200,200], [205,126,74], [20,20,20] ]
colors = [ [104,105,99], [205,209,196], [244,96,54], [60,21,24] ]

class Frame():
    def __init__(self, center = complex(0,0), scale = 0.05):
        self.width = WIDTH
        self.height = HEIGHT
        self.scale_x = self.width*scale
        self.scale_y = self.height*scale
        self.center = center
        self.delta_top_left = complex(-self.scale_x/2,self.scale_y/2)
        self.delta_bottom_right = complex(self.scale_x/2,-self.scale_y/2)
        self.diagonal_vec = self.delta_top_left - self.delta_bottom_right        

    def set_center(self, newpos):
        self.center = newpos
    
    def set_scale(self, newscale): #Scale = How much complex plain the image spans
        self.scale_x = self.width*newscale
        self.scale_y = self.height*newscale
        self.delta_top_left = complex(-self.scale_x/2,self.scale_y/2)
        self.delta_bottom_right = complex(self.scale_x/2,-self.scale_y/2)
        self.diagonal_vec = self.delta_top_left - self.delta_bottom_right

def pol(z,pol_coeffs): #pol of roots
    return(pol_coeffs[4]*z**4 + pol_coeffs[3]*z**3 + pol_coeffs[2]*z**2 + pol_coeffs[1]*z + pol_coeffs[0])

def pol_dev(z,pol_dev_coeffs): # derivative of pol of roots
    return(pol_dev_coeffs[3]*z**3 + pol_dev_coeffs[2]*z**2 + pol_dev_coeffs[1]*z + pol_dev_coeffs[0])

def newton_step(z,pol_coeffs,pol_dev_coeffs):
    z2 = pol_dev(z,pol_dev_coeffs)
    try:
        return(z - ( pol(z,pol_coeffs) / (z2) ) )
    except:
        return(z2)

def z_from_pixel(frame,w,h):
    return(frame.center + frame.delta_top_left - complex(frame.diagonal_vec.real*w/frame.width,frame.diagonal_vec.imag*h/frame.height))

def pixel_from_z(frame,z):
    x = frame.width * (z.real - (frame.center.real + frame.delta_top_left.real)) / frame.scale_x
    y = frame.height * ((frame.center.imag + frame.delta_top_left.imag) - z.imag) / frame.scale_y
    return(x,y)

def color_closest(z,roots):
    dists = []
    for i in roots:
        dists.append(abs(z - i))
    return(colors[np.argmin(dists)])

def draw_fotogram(frame,roots,pol_coeffs,pol_dev_coeffs,archivo):
    img  = Image.new('RGB', (frame.width, frame.height))
    px = img.load()
    for x in range(frame.width):
        for y in range(frame.height):
            z = z_from_pixel(frame,x,y)
            for step in range(100):
                z2 = newton_step(z,pol_coeffs,pol_dev_coeffs)
                if abs(z2 - z) < frame.scale_x/1000:
                    break
                else:
                    z = z2
            c = color_closest(z,roots)
            px[x,y] = tuple(c)
    draw = ImageDraw.Draw(img)
    text = f"s:{frame.scale_x}"
    draw.text((5, 5), text, align ="left") 
    # r = 5
    # for root in roots:
    #     p = pixel_from_z(frame,root)
    #     circle_box = [(p[0]-r, p[1]-r),(p[0]+r, p[1]+r)]
    #     draw.ellipse(circle_box, outline=(255,0,0,255))
    # p = pixel_from_z(frame,sum(roots)/3)
    # circle_box = [(p[0]-r, p[1]-r),(p[0]+r, p[1]+r)]
    # draw.ellipse(circle_box, outline=(0,0,255,255))
    img.save(archivo)

# end = time.time()
# print("Time consumed in working: ",end - start)

args = []

date_string = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
folder_path = f"fotogramas_{date_string}"
os.mkdir(folder_path)

print("Calculating arguments...")
STEP = 5
frames = range(0,480,STEP)
centers = list(np.interp(frames,[0,60,480],[complex(-1,-0.2),complex(-1.12,-0.15),complex(-1.12,-0.29)]))
for tiempo in frames:
    c = centers.pop(0)
    t = 1
    archivo = f"{folder_path}/img_{tiempo:03}.png"
    r1 = -8+3*math.cos(math.pi*t/40) - (3*math.sin(math.pi*t/40)+5)*1j
    r2 = -3*math.cos(math.pi*t/60)+(10j)
    r3 = 10-(6j*(2+math.cos(2*math.pi*t/80)))
    r4 = 8-3*math.cos(math.pi*t/40) - (3*math.sin(math.pi*t/40)+5)*1j
    #frame = Frame(center = 600*t/480 + 600*t/480*1j)
    frame = Frame(center = c, scale=0.05*np.exp(-tiempo/10))
    #roots = np.array([(-2*t)+0j, -5+10j, 10-10j])
    roots = np.array([r1, r2, r3, r4])
    pol_coeffs = P.polyfromroots(roots)
    pol_dev_coeffs = [ pol_coeffs[1], 2*pol_coeffs[2], 3*pol_coeffs[3], 4*pol_coeffs[4] ]
    args.append((frame,roots,pol_coeffs,pol_dev_coeffs,archivo))
print("Done.")
print("Rendering frames...")
with Pool(5) as p:
    p.starmap(draw_fotogram, args)