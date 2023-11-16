#!/usr/bin/env python3

import os
import sys
import time
import logging

import statistics

sys.path = sys.path + ["/usr/local/share/ace/python"]

import ace.syscore
import ace.camera
import ace.filterwheel

# Set the configuration up and establish connections
conn = ace.syscore.AceConnection('localhost')
wheel = ace.filterwheel.FilterWheel(conn, '4k', 'Filter Wheel')
camera = ace.camera.Camera(conn, '4k', 'Camera')
filter = "V"

binning = [1,2,3,4]    # Tested binning regimes are 1x1, 2x2, 3x3, 4x4
amplif = ['Amplifier A', 'Amplifier B', 'Amplifier C', 'Amplifier D', 'Quad Readout']  # Test all possible readout regimes
amplif_alias = ['A', 'B', 'C', 'D', 'Q'] # Aliases used in the naming of files
n_exp = 10               # Number of exposures

cwd = camera.archive_directory  # Working directory. By default it is a current archive directory. Uncomment the next string if you want to use the current working directory
# camera.archive_directory = os.get_cwd()
#wheel.go_to(filter)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(asctime)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ", filename=cwd+"/"+"ARC4K_readout_testing.log")
print("------")
print("This script tests the readout rate of ARC 4K camera")
print("of the Thai National Telescope at Mt Inthanon in different regimes")
print("Author: Eugene Semenko (eugene@narit.or.th)")
print("Last modification: 28 Oct 2023")
print("------")
logging.info("------")
logging.info("This script tests the readout rate of ARC 4K camera")
logging.info("of the Thai National Telescope at Mt Inthanon in different regimes")
logging.info("Author: Eugene Semenko (eugene@narit.or.th)")
logging.info("Last modification: 28 Oct 2023")
logging.info("------")

for ii in range(len(amplif)):
    workamp = amplif_alias[ii]
    print("Select amplifier %s" %(workamp))
    logging.info("Select amplifier %s" %(workamp))

    camera.readout_mode = amplif[ii]
    ampl_dir = "%s" %(workamp)

    for workbin in binning:
        bin_dir = "%s/%sx%s" %(ampl_dir, workbin, workbin)
        data_type = "Bias"

        print("Obtaining bias frames: amplifier %s, binning %sx%s" %(workamp, workbin, workbin))
        logging.info("Obtaining bias frames: amplifier %s, binning %sx%s" %(workamp, workbin, workbin))
        readouttime = []
        for f_count in range(n_exp):
            s = "%s/Bias_%s_%sx%s_%04d.fits" % (bin_dir, workamp, workbin, workbin, f_count+1)
            camera.template = s
            print("Running exposure... ")
            logging.info("Running exposure...")
            start = time.time()
            camera.exposure_type = "BIAS"
            camera.expose(0, ace.camera.exposure_type.BIAS, bin_x=workbin, bin_y=workbin, overscan=0, save=False, block=True)  # EXPOSURE!
            elapsed_time = time.time() - start
            readouttime.append(elapsed_time)
            print("Elapsed time: %.5f seconds" %(elapsed_time))
#            time.sleep(3)
        print("---------------------------------")
        readout_mean = statistics.mean(readouttime)
        readout_std = statistics.stdev(readouttime)
        print("Summary for the amplifier %s, binning %dx%d:" %(workamp, workbin, workbin))
        print("Mean readout time: %.3f±%.3f" %(readout_mean, readout_std))
        logging.info("Summary for the amplifier %s, binning %dx%d:" %(workamp, workbin, workbin))
        logging.info("Mean readout time: %.3f±%.3f" %(readout_mean, readout_std))
logging.info("-----------------------------------")
print("-----------------------------------")
sys.exit(0)
