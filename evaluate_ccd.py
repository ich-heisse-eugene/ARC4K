#!/usr/local/bin/python3.11
"""
Automatic script evaluating CCD parameters based on frame statistics.
This script is designed for work with data taken with the ARC4K camera of
the 2.4-m Thai National Telescope.

Author: Eugene Semenko (eugene@narit.or.th)
Last modification: 16 Nov 2023
"""

import sys
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.exposure import equalize_hist
from skimage.transform import resize
import glob

import warnings
warnings.filterwarnings("ignore")

fontsize = 10
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times Palatino, New Century Schoolbook, Bookman, Computer Modern Roman'
mpl.rcParams['text.usetex'] = True

def read_fits(filein):
    hdu = fits.open(filein)
    data = hdu[0].data
    hdu.close()
    return data

def slice_regions(max_x, max_y, dxy, dphi, bin):
    dxy = dxy // bin; minsep = 4
    mbox = (max_x // (minsep*dxy) // 2)
    if (mbox + 1) * minsep * dxy + dxy // 2 >= max_x//2 or (mbox + 1) * minsep * dxy + dxy // 2 >= max_y//2:
        minsep = minsep - 1
        mbox = (max_x // (minsep*dxy) // 2)
    n_reg = 360 // dphi * mbox + 1
    print(f"Selected {n_reg} regions for analysis: {mbox} areas in {360//dphi} sectors")
    reg = np.zeros((n_reg, 2), dtype=int)
    reg[0,0] = max_x // 2; reg[0, 1] = max_y // 2
    count = 1; curphi = 0
    while count < n_reg:
        for j in range(mbox):
            reg[count, 0] = int((max_x // 2) + (j+1) * minsep * dxy * np.cos(curphi * np.pi / 180.))
            reg[count, 1] = int((max_y // 2) + (j+1) * minsep * dxy * np.sin(curphi * np.pi / 180.))
            count += 1
        curphi = curphi + dphi
    return reg

def show_regions(img, regs, dxy, bin, dirname, hist=True, plot=False):
    if plot:
        print("Displaying regions selected for analysis...")
        dxy = dxy // bin
        if hist == True:
            img_heq = equalize_hist(img, nbins=1024)
        else:
            img_heq = img
        fig = plt.figure(figsize=(8,8), tight_layout=True)
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel("CCD X [pixel]")
        ax.set_ylabel("CCD Y [pixel]")
        ax.imshow(img_heq, cmap="gist_gray", origin="lower", aspect="equal")
        for x in range(regs.shape[0]):
            ax.plot(regs[x, 0], regs[x, 1], color='r', marker='+', ms=10)
            ax.add_patch(mpl.patches.Rectangle((regs[x, 0]-dxy/2, regs[x, 1]-dxy/2), dxy, dxy, fc ='none', ec ='b', lw = 0.8))
            fig.savefig(dirname+"/regions.pdf", dpi=250)
        plt.show()
    return None

def read_bias(amp, bin):
    bin = int(bin)
    workdir = f"{amp}/{bin}x{bin}/"
    pattern = f"Bias_{amp}_{bin}x{bin}_*.fits"
    biaslist = glob.glob(workdir+pattern)
    bias = [read_fits(f) for f in biaslist]
    med = np.median(bias, axis=0)
    mad_std = 1.4826 * np.median(bias - med)
    filtered = (bias - med) / mad_std > 6 # 6*sigma level
    bias = np.ma.array(bias, mask=filtered)
    masterbias = np.mean(bias, axis=0)
    masterbias_fl = masterbias.flatten()
    npix = len(masterbias_fl)
    mn = np.mean(masterbias_fl)
    md = np.median(masterbias_fl)
    stdev = np.std(masterbias_fl)
    min = np.min(masterbias_fl)
    max = np.max(masterbias_fl)
    dim = masterbias.shape
    print(f"Master frame is an average of {len(biaslist)} files")
    print(f"Master bias image size: {dim[0]}x{dim[1]} pixels")
    print(f"Master bias statistics: mean {mn:.3f}, mode {md:.3f}, stdev {stdev:.3f}, max {max:.3f}, min {min:.3f}")
    masterbias = masterbias.filled(fill_value=mn)
    bias = bias.filled(fill_value=mn)
    rms = 2.355*np.std(bias)  # FWHM
    # fig0 = plt.figure(figsize=(8,8), tight_layout=True)
    # ax0 = fig0.add_subplot(1,1,1)
    # ax0.set_xlabel("CCD X [pixel]")
    # ax0.set_ylabel("CCD Y [pixel]")
    # masterbiasheq = equalize_hist(masterbias, nbins=1024)
    # ax0.imshow(masterbias, cmap="gist_gray", origin="lower", aspect="equal", clim=(md-stdev, md+stdev))
    # fig1 = plt.figure(figsize=(6,4), tight_layout=True)
    # ax1 = fig1.add_subplot(1,1,1)
    # ax1.hist(masterbias_fl, bins=256, color='k')
    # ax1.set_xlabel("Signal [ADU]")
    # ax1.set_ylabel("Number")
    # plt.show()
    return masterbias, rms

def parse_data(dirname):
    bias_files = glob.glob(dirname+'/Bias*.fits')
    flat_files = glob.glob(dirname+'/DomeFlat*.fits')
    nbias = len(bias_files)
    if nbias < 3:
        nbias = 0
    exposures = np.unique([float(x.split('_')[1].replace('s', '')) for x in flat_files])
    print(f"Directory \"{dirname}\" with data contains")
    print(f"{nbias} files of bias")
    exposures_out = exposures
    for e in exposures:
        cnt = len([itm for itm in flat_files if itm.find(str(e)+'s') != -1])
        print(f"Texp = {e} s: found {cnt} files")
        if cnt < 3:
            print("Less than 3 files. Excluding...")
            exposures_out.remove(e)
    return nbias, exposures_out

def process_exposure(amp, bin, exptime, masterbias, roi, dxy):
    dxy = dxy // bin
    dirname = amp + '/' + str(bin)+'x'+str(bin)
    flatlist = glob.glob(dirname + f"/DomeFlat_{exptime}s_{amp}_{bin}x{bin}_*.fits")
    flat = [read_fits(f) for f in flatlist]
    masterimage = np.mean(flat, axis=0)
    deviation = flat - masterimage
    del flat
    vm = np.zeros(len(roi)); sav = np.zeros(len(roi))
    for cnt in range(len(roi)):
        Npix = len(masterimage[roi[cnt, 0]-dxy//2: roi[cnt, 0]+dxy//2, roi[cnt, 1]-dxy//2: roi[cnt, 1]+dxy//2].flatten())
        deviation_subframe = deviation[:, roi[cnt, 0]-dxy//2: roi[cnt, 0]+dxy//2, roi[cnt, 1]-dxy//2: roi[cnt,1]+dxy//2]
        masterimage_subframe = masterimage[roi[cnt, 0]-dxy//2: roi[cnt, 0]+dxy//2, roi[cnt, 1]-dxy//2: roi[cnt,1]+dxy//2]
        masterbias_subframe = masterbias[roi[cnt, 0]-dxy//2: roi[cnt, 0]+dxy//2, roi[cnt, 1]-dxy//2: roi[cnt,1]+dxy//2]
        vm[cnt] = np.sum(deviation_subframe**2) / ((len(flatlist)-1)*Npix)
        sav[cnt] = np.sum(masterimage_subframe - masterbias_subframe) / Npix
    return np.mean(vm), np.mean(sav), masterimage - masterbias

def linear_fit(x, b):
    return b + x

def gain_noise(S0, V0, rms, amp, bin, dirname, plot=False):
    nbits = 16
    filt = np.where((S0 >= 0.2*np.max(S0)) & (S0 <= 0.8*np.max(S0)))
    S = S0[filt]; V = V0[filt]
    # Logarithmic
    logS = np.log10(S); logV = np.log10(V)
    minx = 0.9 * np.min(np.log10(S))
    maxx = 1.1 * np.max(np.log10(S))
    miny = 0.9 * np.min(np.log10(V))
    maxy = 1.1 * np.max(np.log10(V))
    popt, pcov = curve_fit(linear_fit, logS, logV)
    perr = np.sqrt(np.diag(pcov))
    g = 10**np.abs(popt[0])
    g_err = np.abs(perr[0]/(np.log(10)*popt[0]))
    n_rd = rms * g
    print(f"g = {g:.4f} ± {g_err:.4f} e-/ADU")
    print(f"n = {n_rd:.1f} e-")
    if plot:
        fig1 = plt.figure(figsize=(8, 6), tight_layout=True)
        ax1 = fig1.add_subplot(1,1,1)
        ax1.set_title(fr"Photon Transfer Curve, Amplifier {amp}, binning {bin}$\times${bin}")
        ax1.set_xlabel(r"$\log_{10}$ (Signal) [ADU]")
        ax1.set_ylabel(r"$\log_{10}$ (Variance) [ADU]")
        ax1.tick_params(axis='both', direction='in')
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')
        ax1.set_xlim(0.95*minx, maxx)
        ax1.set_ylim(0.95*miny, maxy)
        xx1 = np.linspace(minx, np.log10(2**nbits), 100)
        ax1.plot(xx1, linear_fit(xx1, popt[0]), ls='-', color='b', lw=0.5)
        ax1.plot(np.log10(S0), np.log10(V0), color='red', marker='o', ls='', ms=5)
        ax1.plot([np.log10(2**nbits), np.log10(2**nbits)], [0.95*miny, maxy], ls='--', lw=2, color='gray')
        ax1.text(np.log10(2**nbits)+0.01, maxy-0.8, "Saturation", color="gray", rotation=-90, fontsize=fontsize+2)
        ax1.grid(True, axis='both', zorder=1000)
        comment = '\n'.join((
        r'$g=%.2f\,\mathrm{e^{-}/ADU}$' % (g, ),
        r'$n_\mathrm{rd}=%.1f\,\mathrm{e^{-}}$' % (n_rd, )))
        props = dict(boxstyle='round', facecolor='white', lw=1)
        ax1.text(0.78, 0.15, comment, transform=ax1.transAxes, fontsize=fontsize+2, \
                verticalalignment='top', bbox=props)
        fig1.savefig(dirname+"/gain_noise.pdf", dpi=250)
        plt.show()
    return g, n_rd


if __name__ == "__main__":
    amp = ['A', 'B', 'C', 'D']
    bin = [1, 2, 3, 4]
    dxy = 200
    dphi = 45
    for a in amp:
        for b in bin:
            dirname = a+'/'+str(b)+'x'+str(b)
            nbias, exposures = parse_data(dirname)
            nexposures = len(exposures)
            if nbias == 0:
                print("Incomplete data set. Skip...")
                break
            masterbias, rms = read_bias(a, b)
            roi = slice_regions(masterbias.shape[0], masterbias.shape[1], dxy, dphi, b)
            nregions = len(roi)
            vm = np.array([]); sav = np.array([])
            for e_cnt in range(nexposures):
                vmk, savk, _ = process_exposure(a, b, exposures[e_cnt], masterbias, roi, dxy)
                vm = np.append(vm, vmk); sav = np.append(sav, savk)
            _, _, masterflat = process_exposure(a, b, exposures[nexposures//2], masterbias, roi, dxy)
            show_regions(masterflat, roi, dxy, b, dirname, True, False)
            g, n_rd = gain_noise(sav, vm, rms, a, b, dirname, True)


    sys.exit(0)
