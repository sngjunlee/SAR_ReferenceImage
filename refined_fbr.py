import os
path = '/home/dltmd/gdrive/SAR_import_file'

os.chdir(path)

import OriBuri as ob
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import scipy
from alive_progress import alive_bar
import pickle

'''
from os import listdir

data_path = '/home/dltmd/gdrive/capstone/pus_tif/'
data_list = np.sort(listdir(data_path))

grd = []
for i in range (len(data_list)):
    grd.append(ob.getTif(data_path, data_list[i], 'grd'))
    
amp = ob.tiff_match(grd)
for i in range (len(amp)):
    amp[i]['Band'] = amp[i]['Band'][200:600,::]
'''

with open('/home/dltmd/gdrive/capstone/s1a_slc_cal_busan.pkl', 'rb') as f:
    slc = pickle.load(f)

amp = slc.copy()
for i in range (len(slc)):
    amp[i]['Band'] = np.abs(amp[i]['Band'])
    
def getSignal(im_list, a, b):
    signal = []
    for i in range(len(im_list)):
        signal.append(im_list[i][a, b])
    return signal

def generate_x(vvlist):
    x = []
    for i in range (0, len(vvlist)):
        x.append(i)
    return x

def getFBR (product, alpha, enl):
    from math import sqrt, gamma

    stack = [] 
    for i in range (len(product)):
        tmp = product[i]['Band'].copy()
        stack.append(tmp)

    stack = np.array(stack)
    a, b, c = np.shape(stack)
    fbr = np.zeros((b,c))
    fbr_idx = np.zeros((b,c))
    mean_map = np.zeros((b,c))
    std_map = np.zeros((b,c))

    with alive_bar(b, force_tty = True) as bar:
        for i in range (0, b):
            for j in range (0, c):
                signal = getSignal(stack, i, j)
                signal_sort = np.sort(signal)[::-1]
                del signal
                sub_sample = []

                while 1:
                    amp = np.std(signal_sort) / np.mean(signal_sort)
                    cv = sqrt(((gamma(enl) * gamma(enl + 1))
                                /(gamma(enl + 0.5) ** 2)) -1)
                    th = cv + (alpha / (sqrt(len(signal_sort))+0.0000001))
                    if amp < th or len(signal_sort) == 1:
                        fbr[i,j] = np.mean(signal_sort)
                        fbr_idx[i,j] = len(signal_sort)
                        break
                    else:
                        sub_sample.append(signal_sort[0])
                        signal_sort = np.delete(signal_sort, 0)
                        continue
                if len(sub_sample) == 0:
                    continue
                else:
                    mean_map[i,j] = np.mean(sub_sample)
                    std_map[i,j] = np.std(sub_sample)
                
            bar()
    return fbr, fbr_idx, mean_map, std_map

fbr, fbr_idx, mean_map, std_map = getFBR(amp, 1, 0.9) # 0.9

plt.imshow(mean_map, cmap='jet')
plt.clim(0,10)
plt.show()

a,b = np.shape(amp[-1]['Band'])
img = amp[10]['Band']
m_map = np.zeros((a,b))
s_map = np.zeros((a,b))

for i in range (2,a-2):
    for j in range (2,b-2):
        m_map[i,j] = np.mean(img[i-2:i+3, j-2:j+3])
        s_map[i,j] = np.std(img[i-2:i+3, j-2:j+3])

def imgshow_corrcoef (ref_img, img1, img2, clim, title):
     #plt.figure(figsize = (15,15))
    def onclick(event):
        if event.button == 1:
            print(f'pixel coords: x={int(event.xdata)}, y={int(event.ydata)}')
            corrcoef_3 = np.corrcoef(img1[int(event.ydata)-1:int(event.ydata)+2, int(event.xdata)-1:int(event.xdata)+2].flatten(),
                                     img2[int(event.ydata)-1:int(event.ydata)+2, int(event.xdata)-1:int(event.xdata)+2].flatten())
            corrcoef_5 = np.corrcoef(img1[int(event.ydata)-3:int(event.ydata)+4, int(event.xdata)-3:int(event.xdata)+4].flatten(),
                                     img2[int(event.ydata)-3:int(event.ydata)+4, int(event.xdata)-3:int(event.xdata)+4].flatten())
            corrcoef_7 = np.corrcoef(img1[int(event.ydata)-5:int(event.ydata)+6, int(event.xdata)-5:int(event.xdata)+6].flatten(),
                                     img2[int(event.ydata)-5:int(event.ydata)+6, int(event.xdata)-5:int(event.xdata)+6].flatten())
            corrcoef_9 = np.corrcoef(img1[int(event.ydata)-7:int(event.ydata)+8, int(event.xdata)-7:int(event.xdata)+8].flatten(),
                                     img2[int(event.ydata)-7:int(event.ydata)+8, int(event.xdata)-7:int(event.xdata)+8].flatten())
            
            plt.figure()
            plt.subplot(2,2,1)
            plt.plot(img1[int(event.ydata)-1:int(event.ydata)+2, int(event.xdata)-1:int(event.xdata)+2].flatten(),
                     img2[int(event.ydata)-1:int(event.ydata)+2, int(event.xdata)-1:int(event.xdata)+2].flatten(),
                     '.')
            plt.title('3: %f' %(corrcoef_3[1,0]))
            plt.subplot(2,2,2)
            plt.plot(img1[int(event.ydata)-2:int(event.ydata)+3, int(event.xdata)-2:int(event.xdata)+3].flatten(),
                     img2[int(event.ydata)-2:int(event.ydata)+3, int(event.xdata)-2:int(event.xdata)+3].flatten(),
                     '.')
            plt.title('5: %f' %(corrcoef_5[1,0]))
            plt.subplot(2,2,3)
            plt.plot(img1[int(event.ydata)-3:int(event.ydata)+4, int(event.xdata)-3:int(event.xdata)+4].flatten(),
                     img2[int(event.ydata)-3:int(event.ydata)+4, int(event.xdata)-3:int(event.xdata)+4].flatten(),
                     '.')
            plt.title('7: %f' %(corrcoef_7[1,0]))
            plt.subplot(2,2,4)
            plt.plot(img1[int(event.ydata)-4:int(event.ydata)+5, int(event.xdata)-4:int(event.xdata)+5].flatten(),
                     img2[int(event.ydata)-4:int(event.ydata)+5, int(event.xdata)-4:int(event.xdata)+5].flatten(),
                     '.')
            plt.title('9: %f' %(corrcoef_9[1,0]))
            plt.show()
    fig, ax = plt.subplots()
    a = plt.imshow(ref_img, cmap = 'gray')
    a.set_clim(clim)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title(title)
    return plt.show()
 

imgshow_corrcoef(m_map, mean_map, m_map, (0,1), '2')

c = len(amp)
a,b = np.shape(amp[-1]['Band'])
corr_map = np.zeros((a,b))
for i in range (a-7):
    for j in range (b-7):
        win1 = m_map[i:i+7, j:j+7]
        win2 = mean_map[i:i+7, j:j+7]
        corr = np.corrcoef(win1.flatten(), win2.flatten())[0,1]
        if corr >= 0:
            corr_map[i+3,j+3] = corr
        else:
            corr_map[i+3,j+3] = 0
            
c = len(slc)
a,b = np.shape(amp[-1]['Band'])
corr_map = np.zeros((a,b))
corr_maps = []
for k in range (c):
    img = amp[k]['Band'].copy()
    m_map = np.zeros((a,b))
    corr_map = np.zeros((a,b))
    
    for i in range (2,a-2):
        for j in range (2,b-2):
            m_map[i,j] = np.mean(img[i-2:i+3, j-2:j+3])
            #s_map[i,j] = np.std(img[i-2:i+3, j-2:j+3])
            
    for i in range (a-7):
        for j in range (b-7):
            win1 = m_map[i:i+7, j:j+7]
            win2 = mean_map[i:i+7, j:j+7]
            corr = np.corrcoef(win1.flatten(), win2.flatten())[0,1]
            if corr >= 0.6:
                corr_map[i+3,j+3] = corr
            else:
                corr_map[i+3,j+3] = 0
    corr_maps.append(corr_map)
    del corr_map
    print(k+1, 'images completed')
    
import rasterio
with rasterio.open('/mnt/z/coh.tif') as src:
    coh = np.array(src.read(1))

coh = sum(coh)/len(coh)  

with open('/mnt/z/corr_maps.pkl', 'rb') as f:
    corr_maps = pickle.load(f)
    
    
test = amp.copy()
corr = corr_maps.copy()
for i in range (len(outputlist)):
    #corr_maps[i][coh >= 0.5] = 0
    test[i]['Band'][(outputlist[i] >= 0.6)] = 0

   
with open('/mnt/z/s1a_slc_cal_busan.pkl', 'rb') as f:
    slc = pickle.load(f)

amp = slc.copy()
for i in range (len(slc)):
    amp[i]['Band'] = np.abs(amp[i]['Band'])
    
refined_fbr = ob.getFBR(test, 1, 0.9)

plt.figure()
plt.subplot(2,1,1)
plt.imshow(corr[-1], cmap='jet')
plt.clim(0,1)
plt.title('Amplitude Image (2024.03.02)')
plt.subplot(2,1,2)
plt.imshow(test[-1]['Band'], cmap='gray')
plt.clim(0,1)
plt.title('Ship Masked Image (2024.03.02)')
plt.show()

plt.subplot(2,1,1)
plt.imshow(coh, cmap = 'jet')
plt.title('FBR')
plt.clim(0,1)
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(refined_fbr[0], cmap = 'gray')
plt.title('Improved FBR')
plt.clim(0,2)
plt.show()

#%%
import multiprocessing


def mult_corr (img0):
    img = img0['Band'].copy()
    a,b = np.shape(img)
    m_map = np.zeros((a,b))
    corr_map = np.zeros((a,b))
    
    for i in range (2,a-2):
        for j in range (2,b-2):
            m_map[i,j] = np.mean(img[i-2:i+3, j-2:j+3])
            #s_map[i,j] = np.std(img[i-2:i+3, j-2:j+3])
            
    for i in range (0,a,3):
        for j in range (0,b,3):
            win1 = m_map[i:i+3, j:j+3]
            win2 = mean_map[i:i+3, j:j+3]
            corr = np.corrcoef(win1.flatten(), win2.flatten())[0,1]
            if corr >= 0.6:
                #corr_map[i+1,j+1] = corr
                corr_map[i:i+3,j:j+3] = corr
            else:
                #corr_map[i+1,j+1] = 0
                corr_map[i:i+3,j:j+3] = 0
                
    print('images completed')
    return corr_map

if __name__ =='__main__':
    inputlist = amp.copy()
    pool = multiprocessing.Pool(processes=6)
    outputlist = pool.map(mult_corr, inputlist)


test = amp
for i in range (len(outputlist)):
    test[i]['Band'][outputlist[i] >= 0.6] = 0
    
for i in range (len(test)):
    test[i]['Band'] = idw(test[i]['Band'], 51, 'idw')
    
refined_fbr = ob.getFBR(test, 1, 0.9)



plt.subplot(2,1,1)
plt.imshow(test[0]['Band'], cmap = 'gray')
plt.title('FBR')
plt.clim(0,1)
plt.subplot(2,1,2)
plt.imshow(refined_fbr[0], cmap = 'gray')
plt.title('Improved FBR')
plt.clim(0,2)
plt.show()


#%% idw


def generate_dist (size):
    import numpy as np
    
    x = np.zeros((size, size))
    y = np.zeros((size, size))
    
    for i in range (size):
        x[i,::] = np.arange(-(size-1)/2,(size-1)/2+1)
        y[::,i] = np.arange(-(size-1)/2,(size-1)/2+1)
        
    dist = np.sqrt(x**2+y**2)
    
    return dist

def idw (img, size, type):
    
    import numpy as np
    
    dist = generate_dist(size)
    dist[dist == 0] = -1
    weight = 1/dist
    weight[dist == -1] = 0
    fcal = img.copy()
    a, b = np.shape(img)
    for i in range (a - size):
        for j in range (b - size):
            if img[i + int((size-1)/2), j + int((size-1)/2)] == 0:
                if type == 'idw':
                    weight_cal = (weight*fcal[i:i+size, j:j+size])
                    weight_cal[weight_cal >0] = 1 
                    #img[i + int((size-1)/2), j + int((size-1)/2)] = np.sum(np.arctan(fcal[i:i+size, j:j+size]) * (weight)) / np.sum(weight)
                    img[i + int((size-1)/2), j + int((size-1)/2)] = np.sum(fcal[i:i+size, j:j+size] * (weight)) / np.sum(weight)
                if type == 'min':
                    tmp = fcal[i:i+size, j:j+size]
                    tmp[tmp >= 0.2] = 0
                    img[i + int((size-1)/2), j + int((size-1)/2)] = np.max(tmp)

            else:
                continue
            
    return img


a,b = np.shape(refined_fbr[0])
coord = []
z = []
x = []
y = []
for i in range (a):
    for j in range (b):
        if refined_fbr[0][i,j] == 0:
            x.append(j)
            y.append(i)
            continue


for i in range (len(x)):
    refined_fbr[0][y[i]-1:y[i]+2,x[i]-1:x[i]+2] = 0
        

result = refined_fbr[0].copy()
result[result == 0] = 0.15
ob.getHist(refined_fbr[0], np.arange(0,1,0.01))

result = idw(refined_fbr[0].copy(), 51, 'idw')
rgb = ob.getRGB(fbr, ob.lee_filter(amp[-1], 5)['Band'], ob.lee_filter(amp[-1], 5)['Band'])
rgb2 = ob.getRGB(refined_fbr[0], ob.lee_filter(amp[-1], 5)['Band'], ob.lee_filter(amp[-1], 5)['Band'])


plt.figure()
plt.subplot(2,2,1)
plt.imshow(rgb, cmap='gray')
plt.title('RGB with FBR')
plt.clim(0,1)
plt.subplot(2,2,3)
plt.imshow(rgb2, cmap='gray')
plt.title('RGB with New FBR')
plt.clim(0,1)
plt.subplot(2,2,2)
plt.imshow(fbr, cmap='gray')
plt.clim(0,1)
plt.title('FBR')
plt.subplot(2,2,4)
plt.imshow(result, cmap='gray')
plt.clim(0,1)
plt.title('New FBR')
plt.show()

#%% interpolation
from sklearn import cluster
param = cluster.fit(input_data)

cl.labels_

img_cl = cl.labels_.reshape((a,b))



def get_idx(data):
    a,b = np.shape(data)
    
    x = []
    y = []

    for i in range (a):
        for j in range (b):
            if data[i,j] == 1:
                x.append(j)
                y.append(i)

            
    return x, y

x, y, z = get_idx(ratio2)

X = np.zeros((len(x), 2))
X[:,0] = x
X[:,1] = y
cl = cluster.DBSCAN(eps=0.5, min_samples=3).fit(X)
plt.scatter(x,y,c=cl.labels_,s=1,cmap='tab20c')
plt.show()

rgb = ob.getRGB(fbr, amp[-1]['Band'], amp[-1]['Band'])
rgb2 = ob.getRGB(result, amp[-1]['Band'], amp[-1]['Band'])

plt.figure()
plt.subplot(2,2,1)
plt.imshow(fbr, cmap='gray')
plt.title('FBR')
plt.clim(0,1)
plt.subplot(2,2,3)
plt.imshow(rgb2, cmap='gray')
plt.title('RGB with New FBR')
plt.clim(0,1)
plt.subplot(2,2,2)
plt.imshow(refined_fbr[0], cmap='gray')
plt.clim(0,1)
plt.title('before interpolation')
plt.subplot(2,2,4)
plt.imshow(result, cmap='gray')
plt.clim(0,2)
plt.title('New FBR')
plt.show()

ratio1 = amp[-1]['Band']/fbr
ratio2 = amp[-1]['Band']/result
ratio1[ratio1 == 0] = 0.0000001
ratio2[ratio2 == 0] = 0.0000001

ships1 = ob.cfar_db({'Band': ratio1}, 0.05, 51)
ships2 = ob.cfar_db({'Band': ratio2}, 0.05, 51)

from scipy.ndimage import median_filter
median1 = median_filter(ships1['Band'], size=5)
median2 = median_filter(ships2['Band'], size=5)

plt.figure()
plt.subplot(2,1,1)
plt.imshow(amp[-1]['Band'], cmap = 'gray')
plt.clim(0,1)
plt.imshow(median1, cmap = 'bwr', alpha=0.5)
plt.clim(-1,1)
plt.title('fbr')

plt.subplot(2,1,2)
plt.imshow(amp[-1]['Band'], cmap = 'gray')
plt.clim(0,1)
plt.imshow(median2, cmap = 'bwr', alpha=0.5)
plt.clim(-1,1)
plt.title('improved fbr')
plt.show()

x, y = get_idx(median1)

plt.figure()
plt.subplot(2,1,1)
plt.imshow(amp[-1]['Band'], cmap = 'gray')
plt.clim(0,1)
plt.scatter(x, y, c = 'g', s=1)
plt.clim(-1,1)
plt.title('fbr')

x, y = get_idx(median2)

plt.subplot(2,1,2)
plt.imshow(amp[-1]['Band'], cmap = 'gray')
plt.clim(0,1)
plt.scatter(x, y, c = 'r', s=1)
plt.clim(-1,1)
plt.title('improved fbr')
plt.show()





































