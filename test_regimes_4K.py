#!/usr/bin/env python3

import os
import sys
import time
import logging

sys.path = sys.path + ["/usr/local/share/ace/python"]

import ace.syscore
import ace.camera
import ace.filterwheel

start = time.process_time()

# Set the configuration up and establish connections
conn = ace.syscore.AceConnection('localhost')
wheel = ace.filterwheel.FilterWheel(conn, '4k', 'Filter Wheel')
camera = ace.camera.Camera(conn, '4k', 'Camera')

target_name = ["Bias", "DomeFlat"]  # Target names used in file naming
filter = "V"      # Testing is performed only in V filter using the dome flats
binning = [1, 2, 3, 4]    # Tested binning regimes are 1x1, 2x2, 3x3, 4x4
amplif = ['Amplifier A', 'Amplifier B', 'Amplifier C', 'Amplifier D', 'Quad Readout']  # Test all possible readout regimes
amplif_alias = ['A', 'B', 'C', 'D', 'Q'] # Aliases used in the naming of files
texp_min = 0.1            # Minimum possible exposure time for all regimes
texp_max = [45, 11, 5, 2.5] # Maximum exposure time when the maximum counts are close to saturation. MUST be specified in advance
n_texp = 10               # Number of exposure between texp_min and texp_max
n_bias = 5               # Number of bias frames. MUST be an odd number
n_flat = 5                # Number of flat field images. MUST be an odd number

cwd = camera.archive_directory  # Working directory. By default it is a current archive directory. Uncomment the next string if you want to use the current working directory
# camera.archive_directory = os.get_cwd()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(asctime)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ", filename=cwd+"/"+"ARC4K_testing.log")
print("------")
print("This script runs a series of tests for various working regimes of the ARC 4K camera")
print("of the Thai National Telescope at Mt Inthanon")
print("Author: Eugene Semenko (eugene@narit.or.th)")
print("Last modification: 27 Oct 2023")
print("------")
logging.info("------")
logging.info("This script runs a series of tests for various working regimes of the ARC 4K camera")
logging.info("of the Thai National Telescope at Mt Inthanon")
logging.info("Author: Eugene Semenko (eugene@narit.or.th)")
logging.info("Last modification: 27 Oct 2023")
logging.info("------")

print(" --- Setting up the environment ---")
print("Moving filter wheel to: %s ..." %(filter))
logging.info(" --- Setting up the environment ---")
logging.info("Moving filter wheel to: %s..." %(filter))

wheel.go_to(filter)  # Go!

print("done")
logging.info("...done")

for ii in range(len(amplif)):
    workamp = amplif_alias[ii]
    print("Select amplifier %s" %(workamp))
    logging.info("Select amplifier %s" %(workamp))

    camera.readout_mode = amplif[ii]
    ampl_dir = "%s" %(workamp)

    for workbin, worktexp_max in zip(binning, texp_max):
        bin_dir = "%s/%sx%s" %(ampl_dir, workbin, workbin)

        data_type = "Bias"

        print("Obtaining bias frames: amplifier %s, binning %sx%s" %(workamp, workbin, workbin))
        logging.info("Obtaining bias frames: amplifier %s, binning %sx%s" %(workamp, workbin, workbin))
        for f_count in range(n_bias):
            s = "%s/%s_%s_%sx%s_%04d.fits" % (bin_dir, target_name[0], workamp, workbin, workbin, f_count+1)
            print("Setting image filename template to: %s" %(s))
            logging.info("Setting image filename template to: %s" %(s))
            camera.template = s
            print("Running exposure... ")
            logging.info("Running exposure...")
            camera.expose(0, ace.camera.exposure_type.BIAS, bin_x=workbin, bin_y=workbin, overscan=0, save=True, block=True)  # EXPOSURE!
            print("done")
            logging.info("...done")

        texp = [texp_min + x*(worktexp_max-texp_min)/n_texp for x in range(n_texp+1)]
        for worktexp in texp:
            worktexp = "%.1f" %(worktexp)
            print("Obtaining flat frames: amplifier %s, binning %sx%s, Texp = %s" %(workamp, workbin, workbin, worktexp))
            logging.info("Obtaining flat frames: amplifier %s, binning %sx%s, Texp = %s" %(workamp, workbin, workbin, worktexp))
            for f_count in range(n_flat):
                s = "%s/%s_%ss_%s_%sx%s_%04d.fits" % (bin_dir, target_name[1], worktexp, workamp, workbin, workbin, f_count+1)
                print("Setting image filename template to: %s" %(s))
                logging.info("Setting image filename template to: %s" %(s))
                camera.template = s
                print("Running exposure... ")
                logging.info("Running exposure...")
                camera.expose(float(worktexp), ace.camera.exposure_type.FLAT, bin_x=workbin, bin_y=workbin, overscan=0, save=True, block=True) # EXPOSURE!
                print("done")
                logging.info("...done")
        print("---")
elapsed_time = time.process_time() - start
print("Testing took %f seconds" %(elapsed_time))
print("-----------------------------------")
logging.info("Testing took %f seconds" %(elapsed_time))
logging.info("-----------------------------------")
sys.exit(0)
