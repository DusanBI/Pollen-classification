#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np
import json
from math import factorial
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
import cv2


# In[2]:


class preprocess:
    def normalize_image(opd, cut, normalize=False, smooth=True):
        if len(opd['Scattering']) > 3:
            image = np.asarray(opd['Scattering'])
        else:
            image = np.asarray(opd['Scattering']['Image'])
        image = np.reshape(image, [-1, 24])
        im = np.zeros([2000, 20])
        cut = int(cut)
        im_x = np.sum(image, axis=1) / 256
        N = len(im_x)
        if N < 450:
            cm_x = 0
            for i in range(N):
                cm_x += im_x[i] * i
            cm_x /= im_x.sum()
            cm_x = int(cm_x)
            im[1000 - cm_x:1000 + (N - cm_x), :] = image[:, 2:22]
            im = im[1000 - cut:1000 + cut, :]
            if smooth:
                for i in range(20):
                    im[:, i] = preprocess.savitzky_golay(im[:, i] ** 0.5, 5, 3)
            # im[:,0:2] = 0
            # im[:,22:24] = 0
            im = np.transpose(im)
            if normalize:
                return np.asarray(im / im.sum())
            else:
                return np.asarray(im)

    def __fit_exp_decay__(x, a, b, c):
        x = np.array(x, dtype=np.float64)
        return np.array(a * np.exp(-x / b) + c, dtype=np.float64)

    def __fit_exp_approx_decay__(x, a, b, c):
        x = np.array(x, dtype=np.float64)
        return np.array(a * (1.0 - (1 / b) * x + 1 / (b * 2.0) * x ** 2 - 1 / (b * 6.0) * x ** 3 + 1 / (
                b * 24.0) * x ** 4 - 1 / (b * 120.0) * x ** 5 + 1 / (b * 720.0) * x ** 6) + c, dtype=np.float64)

    # reconstruction of liti per band
    def __reconstruct_liti__(liti):

        indices_max = np.argwhere(liti == np.amax(liti))

        if np.amax(liti) == 4088:

            if len(indices_max) > 1:
                ind_f = indices_max[0][0]
                ind_l = indices_max[-1][0]

                if (ind_f < 10) or (ind_l > (len(liti) - 10)):
                    liti[0::] = 0
                    return liti
                else:

                    # gauss_extrapolation
                    x1 = np.arange(0, ind_f + 1)
                    s1 = InterpolatedUnivariateSpline(x1, liti[0:ind_f + 1].astype("float64"), k=2)
                    gauss_exc_extrapolate = s1(np.arange(ind_f, ind_l + 1))

                    # exp_extrapolation
                    x2 = np.arange(0, len(liti) - ind_l)
                    popt, pcov = curve_fit(preprocess.__fit_exp_approx_decay__, x2 + (ind_l - ind_f + 1),
                                           liti[ind_l::].astype("float64"), maxfev=1000)
                    exp_decay_extrapolate = preprocess.__fit_exp_approx_decay__(np.arange(0, ind_l - ind_f + 1),
                                                                                popt[0], popt[1], popt[2])

                    # signal creation
                    ind_cross = np.argmin(np.abs(exp_decay_extrapolate - gauss_exc_extrapolate))
                    middle_signal = np.concatenate(
                        (gauss_exc_extrapolate[0:ind_cross], exp_decay_extrapolate[ind_cross::]))
                    liti_new = np.concatenate((liti[0:ind_f], middle_signal, liti[ind_l + 1::]))

                    return liti_new
            else:
                return liti
        else:
            return liti

    def normalize_lifitime(opd, normalize=True):
        liti = np.asarray(opd['Lifetime']).reshape(-1, 64)
        counter = 0

        for i in range(liti.shape[0]):
            if np.amax(liti[i]) == 4088:
                liti[i] = preprocess.__reconstruct_liti__(liti[i])

            if (liti[i] == 0).all():
                counter += 1

        if counter == 0:
            lt_im = np.zeros((4, 24))
            liti_low = np.sum(liti, axis=0)
            maxx = np.max(liti_low)
            ind = np.argmax(liti_low)

            if (ind > 10) and (ind < 44):
                lt_im[:, :] = liti[:, ind - 4:20 + ind]
                weights = []
                for i in range(4):
                    weights.append(np.sum(liti[i, ind - 4:12 + ind]) - np.sum(liti[i, 0:16]))
                B = np.asarray(weights)
                A = lt_im
                if normalize:
                    if (maxx > 0) and (B.max() > 0):
                        return A / maxx, B / B.max()
                else:
                    return A, B

    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')

    def spec_correction(opd, normalize=True):
        spec = np.asarray(opd['Spectrometer'])
        spec_2D = spec.reshape(-1, 8)

        b = 0
        if (spec_2D[:, 1] > 20000).any():
            res_spec = spec_2D[:, 1:5]
            b = 1
        else:
            res_spec = spec_2D[:, 0:4]

        if (np.argsort(res_spec[:, 0])[-4:] > 3).all() and (np.argsort(res_spec[:, 0])[-4:] < 10).all():

            if b == 0:
                res_spec = spec_2D[:, 1:5]

            for i in range(res_spec.shape[1]):
                res_spec[:, i] -= np.minimum(spec_2D[:, 6], spec_2D[:, 7])

            for i in range(4):
                res_spec[:, i] = preprocess.savitzky_golay(res_spec[:, i], 5, 3)  # Spectrum is smoothed
            res_spec = np.transpose(res_spec)
            if normalize == True:
                A = res_spec
                if (A.max() > 0):
                    return A / A.max()
            else:
                return res_spec


# In[3]:


# load data into a list
def import_data(path):
    os.chdir(path)
    files = sorted(os.listdir())
    print([files[i].split(".")[0] for i in range(len(files))])
    data = [[], [], []]

    for file in files:
        raw_data = json.loads(open(file).read())
        print(file.split(".")[0])
        print("Num of samples before filtering: ", len(raw_data))
        data_list = [[], [], []]

        for i in range(len(raw_data)):
            if np.max(raw_data[i]["Spectrometer"]) > 2500:  # preprocessing
                scat = preprocess.normalize_image(raw_data[i], cut=60, normalize=False, smooth=True)
                spec = preprocess.spec_correction(raw_data[i], normalize=True)
                life1 = preprocess.normalize_lifitime(raw_data[i], normalize=True)

                if scat is not None and spec is not None and life1 is not None:
                    data_list[0].append(scat)
                    data_list[1].append(spec)
                    data_list[2].append(life1[0])

        print("Num of samples after filtering: ", len(data_list[0]))
        for p in range(len(data)):
            data[p].append(data_list[p])

    return data


# In[4]:


data = import_data('C:\\Users\\Luka\\Dropbox\\My PC (DESKTOP-E57AO1D)\\Desktop\\Dusan pollen\\projekat\\pollen\\train\\train')


# In[5]:


def data_and_labels(p, t):
    os.chdir(p)
    files = [os.path.splitext(filename)[0] for filename in os.listdir(p)]
    d0 = []
    d1 = []
    d2 = []
    l = []
    for i in range(8):
        for j in range(len(t[0][i])):
            d0.append(t[0][i][j])
            d1.append(t[1][i][j])
            d2.append(t[2][i][j])
            l.append(files[i])
    for i in range(8):
        for j in range(len(t[0][8+i])):
            d0.append(t[0][8+i][j])
            d1.append(t[1][8+i][j])
            d2.append(t[2][8+i][j])
            l.append(files[i])
    d0 = np.asarray(d0)
    d1 = np.asarray(d1)
    d2 = np.asarray(d2)
    l = np.asarray(l)
    return d0, d1, d2, l


# In[6]:


path = 'C:\\Users\\Luka\\Dropbox\\My PC (DESKTOP-E57AO1D)\\Desktop\\Dusan pollen\\projekat\\pollen\\train\\train'
keys = ['scat', 'spec', 'life', 'labels']
dataNew = {}
dataNew['scat'],dataNew['spec'], dataNew['life'], dataNew['labels'] = data_and_labels(path, data)


# In[7]:


dataNew


# In[10]:


def export_data_and_labels(p):
    os.chdir(p)

    for key in keys:
        open_file = open('data_' + key + '.pkl', "wb")
        pickle.dump(dataNew[key], open_file)
        open_file.close()


# In[11]:


export_data_and_labels(path)


# In[ ]:




