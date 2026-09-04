#-------------------------------------------------------------------------------------------------------------#
##-----##-----##-----##-----##-----##-----##-----## IMPORTS ##-----##-----##-----##-----##-----##-----##-----##
#-------------------------------------------------------------------------------------------------------------#

# Add the directory where starships' directory is located
from sys import path
path.append('/home/mathisb/Github/starships')

# Add the directory containing the input data (opacity, abundance and stellar specs files)
import os
os.environ['pRT_input_data_path'] = '/home/mathisb/projects/def-rdoyon/shared/Models/petitRADTRANS/input_data/'

from pathlib import Path
from importlib import reload
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
import logging
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm, set_loglevel
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from scipy.interpolate import interp1d

from astropy import stats
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy.modeling.physical_models import BlackBody as bb

# STARSHIPS
import starships.planet_obs as pl_obs
from starships.planet_obs import Observations, Planet

import starships.correlation_class as cc
from starships.correlation_class import Correlations
import starships.correlation as corr

import starships.homemade as hm
import starships.plotting_fcts as pf
import starships.spectrum as spectrum
from starships.spectrum import RotKerTransitCloudy
from starships.mask_tools import interp1d_masked
import starships.petitradtrans_utils as prt  # petitRADTRANS

from petitRADTRANS.physics import guillot_global
from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc

interp1d_masked.iprint=False

import pipeline.reduction as red







#--------------------------------------------------------------------------------------------------------------------------#
##-----##-----##-----##-----##-----##-----##-----## PARAMETERS TO CHANGE ##-----##-----##-----##-----##-----##-----##-----##
#--------------------------------------------------------------------------------------------------------------------------#

## MODEL PARAMETERS ##

pl_name = "TRAPPIST-1 e"

###############################
mole_abundances = [1]  # greenhouse gas abundance
# 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1
cloud = [100]  # opaque cloud deck pressure in bar (None for no clouds)
# 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100

only_specified = True  # only run the injections, reductions and cross-correlations for the values specified above
###############################

# Greenhouse gas
gg = 'CH4'

# Scenario (background gas)
scenario = 'H2rich'  # H2rich (N2rich not implemented yet)

# Pressure
limP = [-10, 2]  # log10 of pressure at top-of-atmosphere and at surface (unit: bar)
P0 = 100  # reference pressure, where R = R_pl (unit: bar)

# Dissociation
dissociation = True  # allow for molecules to break up in the atmosphere

# Wavelength range of the model
wl_range = [0.95, 1.96]  # wavelength range of NIRPS (unit: micrometers)

# Resolution of the model
instr_res = 75000  # resolution of NIRPS

# Type of observation
kind_trans = 'transmission'  # transmission or emission



## INJECTION, REDUCTION AND CROSS-CORRELATION PARAMETERS ##

instrument = "NIRPS_Apr2026"

scaling = 1  # scaling factor to multiply the model (transit depth) with
RV_inj = -50  # RV where to inject the signal

RVsys  = -52.003101*u.km/u.s  # RV of star system, source: Simbad

# Where to inject a signal
nightlist = [
    # NIRPS b
            # '2023-09-07',
            # '2024-07-05',
            # '2024-07-11b',
    # NIRPS d
            '2023-06-08',
            # '2023-08-12',
            # '2023-08-24',
            # '2023-10-20',
    # NIRPS e
            # '2023-07-17',
            # '2023-11-04',
            # '2024-07-11e',
    # NIRPS f
            # '2023-08-28',
            # '2024-11-02',
            # '2025-08-06',
    # NIRPS g
            # '2023-10-05',
            # '2023-11-11',
            # '2025-07-08',
               ]

# Reduction params
nPC = 0  # number of principal components to remove
tellu_frac = 0.40  # telluric fraction to mask (usually varied between 0.2 and 0.5)
mask_wings = 0.97  # fraction of wings of deep tellurics to mask

coeffs = [0.02703969,  1.10037972, -0.96372403,  0.28750393]  # limb darkening coefficients
ld_model = 'nonlinear'

# Define cross-correlation axes
step_RV_inj = 1.0
corr_xlim = 100  # correlation will be done with Vrad values in [-corr_xlim, corr_xlim]

# What to plot
plot_KpVrad = True       # plot Kp/Vrad plot (sigma-clipping) and store the results in a dict
plot_all_orders = False  # plot the correlation map for each order

common_scale = False  # in plot_all_orders, plot all orders with a common scale

# Criteria to exclude an order
max_masked_ratio = 0.9  # excludes the orders where the ratio of masked data points is too high
min_SNR = 5             # excludes the orders where the instrument SNR is too low
min_model_ppm = 5       # excludes the orders where the model atmosphere is close to 0 ppm in transit depth

# Additional orders to exclude
#CH4
orders_to_delete = {'2023-09-07': [33,35,40], '2024-07-05': [68],
                    '2023-08-12': [0,1,2,3,4,5,6,8,9,10,12,59,68], '2023-08-24': [1,5,6,9,33,40], '2023-10-20': [4],
                    '2023-11-04': [11],
                    '2023-08-28': [13,16], '2024-11-02': [41,68],
                    '2023-10-05': [21], '2023-11-11': [25,53],
                   }
#H2O
# orders_to_delete = {'2023-09-07': [33,35,40], '2024-07-05': [68],
#                     '2023-08-12': [0,1,2,3,4,5,6,8,9,10,12,59,68], '2023-08-24': [1,5,6,9,33,40], '2023-10-20': [4],
#                     '2023-11-04': [11],
#                     '2023-08-28': [13], '2024-11-02': [41,68],
#                     '2023-10-05': [21], '2023-11-11': [25,53],
#                    }

# Choose Kp (y-axis) values for the plot
kp_grid = np.linspace(-1000, 1000, 400)

# Set sigma limit and max number of iterations for the sigma-clipping method
clip_sigma = 2
clip_iter = 9







#-------------------------------------------------------------------------------------------------------------------#
##-----##-----##-----##-----##-----##-----##-----## CREATE MODELS ##-----##-----##-----##-----##-----##-----##-----##
#-------------------------------------------------------------------------------------------------------------------#

# Load in the planetary parameters
if pl_name == 'TRAPPIST-1 b':
    ap     = 0.01154*u.au        # semi-major axis of planet
    R_pl   = 1.116*u.R_earth     # radius of planet
    R_star = 0.1192*u.R_sun      # radius of star
    M_star = 0.0898*const.M_sun  # mass of star
    e      = 0                   # eccentricity
    w      = 4.712389*u.rad      # argument of periapsis (equal to 270 degrees)
    T_eq   = 397.6               # equilibrium temperature

elif pl_name == 'TRAPPIST-1 d':
    ap     = 0.02227*u.au        # semi-major axis of planet
    R_pl   = 0.788*u.R_earth     # radius of planet
    R_star = 0.1192*u.R_sun      # radius of star
    M_star = 0.0898*const.M_sun  # mass of star
    e      = 0                   # eccentricity
    w      = 4.712389*u.rad      # argument of periapsis (equal to 270 degrees)
    T_eq   = 286.2               # equilibrium temperature

elif pl_name == 'TRAPPIST-1 e':
    ap     = 0.02925*u.au        # semi-major axis of planet
    R_pl   = 0.920*u.R_earth     # radius of planet
    R_star = 0.1192*u.R_sun      # radius of star
    M_star = 0.0898*const.M_sun  # mass of star
    e      = 0                   # eccentricity
    w      = 4.712389*u.rad      # argument of periapsis (equal to 270 degrees)
    T_eq   = 249.7               # equilibrium temperature

elif pl_name == 'TRAPPIST-1 f':
    ap     = 0.03849*u.au        # semi-major axis of planet
    R_pl   = 1.045*u.R_earth     # radius of planet
    R_star = 0.1192*u.R_sun      # radius of star
    M_star = 0.0898*const.M_sun  # mass of star
    e      = 0                   # eccentricity
    w      = 4.712389*u.rad      # argument of periapsis (equal to 270 degrees)
    T_eq   = 217.7               # equilibrium temperature

elif pl_name == 'TRAPPIST-1 g':
    ap     = 0.04683*u.au        # semi-major axis of planet
    R_pl   = 1.129*u.R_earth     # radius of planet
    R_star = 0.1192*u.R_sun      # radius of star
    M_star = 0.0898*const.M_sun  # mass of star
    e      = 0                   # eccentricity
    w      = 4.712389*u.rad      # argument of periapsis (equal to 270 degrees)
    T_eq   = 197.3               # equilibrium temperature

else:
    raise ValueError(f"Please add the params for planet {pl_name}.")

pl_kwargs = {'M_star': M_star, 
             'R_star': R_star,
             'ap': ap,
             'R_pl': R_pl,
             'excent': e,
             'w': w}

planet_obj = Planet(pl_name, **pl_kwargs)

# Names of the linelists that will be fetched by petitRADTRANS
mols_linelists = {
        'H2O': 'H2O_main_iso',
        'CO': 'CO_all_iso',
        'CO2': 'CO2_main_iso',
        'FeH': 'FeH_main_iso',
        'C2H2': 'C2H2_main_iso',
        'CH4': 'CH4_main_iso',
        'HCN': 'HCN_main_iso',
        'NH3': 'NH3_main_iso',
        'TiO': 'TiO_all_iso',
        'SiO': 'SiO_main_iso',
        'VO': 'VO',
        'OH': 'OH',
        'Na': 'Na',
        'K': 'K',
        'H-': 'H-',
        'H': 'H',
        'e-': 'e-',
        'Al': 'Al',
        'B': 'B',
        'Be': 'Be',
        'Ca': 'Ca',
        'CaII': 'Ca+',
        'Cr': 'Cr',
        'Fe': 'Fe',
        'FeII': 'Fe+',
        'Li': 'Li',
        'Mg': 'Mg',
        'MgII': 'Mg+',
        'N': 'N',
        'Si': 'Si',
        'Ti': 'Ti',
        'V': 'V',
        'VII': 'V+',
        'Y': 'Y',
                 }

if scenario == "N2rich":
    continuum = ['N2-N2']      # continuum absorber species
    rayleigh_species = ['N2']  # species for rayleigh scattering
else:
    continuum = []         # H2-H2 and H2-He only
    rayleigh_species = []  # H2 and He only

gravity = planet_obj.gp.cgs.value  # planetary surface gravity (unit: cm/s^2)

models_shortname = f"{pl_name[-1]}_{scenario}_{gg}"
model_dir = Path('/home/mathisb/scratch/HRS_models/' + models_shortname)  # model directory



for ab in mole_abundances:
    for p_cl in cloud:

        # Model file name
        model_name = Path(f"prt_model_{pl_name.replace(' ', '')}_{scenario}_{gg}_{ab}_pcloud_{p_cl}.npz")
        model_fullpath = model_dir / model_name

        # Check that this model has not already been generated
        if model_fullpath.exists():
            print(f"Model already generated: {model_name}")
            continue

        # Update 100% abundances so they can be taken in by the code
        ab_model = 0.9999999 if ab == 1 else ab
        
        # Add background gas
        if scenario == "H2rich":
            atm_fill = 0.99999999 - ab_model
            H2_ab = 0.85 * atm_fill
            He_ab = 0.15 * atm_fill
            molecules = {'H2': H2_ab, 'He': He_ab}
        else:
            raise ValueError(f"Scenario {scenario} has not been implemented yet.")

        # Add greenhouse gas abundance
        molecules[gg] = ab_model

        # Species required by the model to compute properly
        special_species = {"H", "H2", "H-", "He", "e-"}
        special_abundances = {key: molecules.pop(key) for key in special_species if key in molecules} # extract these species to add them in later
        
        # Generate the atmosphere
        atmos_high, pressures = prt.gen_atm_all([mols_linelists[key] for key in molecules],
                                                limP=limP,
                                                n_pts=50,  # number of pressure values used to compute the atmosphere
                                                mode='lbl',  # high resolution mode with line-by-line treatment; for low-resolution mode, use 'c-k'
                                                lbl_opacity_sampling=4,  # samples one of every nth point of the opacities
                                                continuum_opacities=continuum,
                                                rayleigh_species=rayleigh_species,
                                                wl_range=wl_range)

        # Temperature-pressure profile
        temperatures_day = np.full(len(pressures), T_eq)  # constant temperature throughout the atmosphere

        # Merge user-defined and required species, giving priority to user-specified abundances
        molecules_full = molecules | {species: special_abundances.get(species, 1e-60) for species in special_species}
        
        # Dictionary with the names of the linelists and the abundances
        species_high = {mols_linelists[mol] if mol in mols_linelists else mol: abundance for mol, abundance in molecules_full.items()}
        
        # Calculate mean molecular weight of the atmosphere
        MMW = prt.calc_MMW3(species_high) * np.ones_like(pressures)

        # Generate the model
        wlen, flux_lambda = prt.retrieval_model_plain(atmos_high,
                                                      species_high,
                                                      planet_obj,
                                                      pressures,
                                                      temperatures_day,
                                                      gravity,
                                                      P0,
                                                      p_cl,
                                                      planet_obj.R_pl.cgs.value,
                                                      planet_obj.R_star.cgs.value,
                                                      dissociation=dissociation,
                                                      plot_abundance=False,  # plot the abundances
                                                      kind_trans=kind_trans)

        wave_mod, model_spec = prt.prepare_model(wlen,             # model wavelengths (unit: micrometers)
                                                 flux_lambda,      # transit radius (unit: Earth radii)
                                                 250000,           # full resolution of the model
                                                 Raf=instr_res,    # resampled resolution
                                                 rot_params=None)  # rotational kernel for convolution

        # Create directory and save model
        model_dir.mkdir(parents=True, exist_ok=True)
        np.savez(model_fullpath, wave_mod=wave_mod, model_spec=model_spec, MMW=MMW[0])
        print(f"Model saved: {model_fullpath}")

print("=== ALL MODELS GENERATED ===\n")







#------------------------------------------------------------------------------------------------------------------------------------------------#
##-----##-----##-----##-----##-----##-----##-----## INJECTION, REDUCTION AND CROSS-CORRELATION ##-----##-----##-----##-----##-----##-----##-----##
#------------------------------------------------------------------------------------------------------------------------------------------------#

def inj_rec(config_dict, p, obs, visit_name, wave_mod, mod_spec, plot_model=False, save_dir=None, path_fig=None, debug=False):

    ''' Takes as input a model and a raw data file to inject the model into at a given RV.
    Assumed a list of t.fits files for the night have already been made (so, that split nights has been run)
    
    wave_mod and mod_spec: outputs of petitradtrans model -- mod_spec should be in units of transit depth (R_pl/R_star)**2
    '''
    
    # Where to save the injected data files
    if save_dir == None:
        save_dir = Path(os.environ["SCRATCH"])
        save_dir /= Path(f'HRS_reductions/{config_dict["instrument"]}/{"".join(p.name.split())}/injected_data/{visit_name}/{config_dict["reduction"]}')
        save_dir.mkdir(parents=True, exist_ok=True)
    
    if path_fig == None:
        path_fig = save_dir / 'Figures/'
        path_fig.mkdir(parents=True, exist_ok=True)

    # Check if this injection has already been done (if so, skip it)
    if (save_dir / "list_tcorr.txt").is_file():
        print(f"Injected data already exists: {visit_name}/{config_dict['reduction']}")
        return
    else:
        print(f"Injecting {visit_name} with {config_dict['reduction'][9:]}...")
    
    # Getting the exposures in and out of transit
    obs.calc_sequence(plot=False)
    in_transit = obs.iIn
    out_transit = obs.iOut

    # Getting the lightcurve
    exp_times = obs.t
    lightcurve = obs.alpha

    # Create the window function to scale the model 
    Wc = lightcurve / np.max(lightcurve)

    # Getting velocities for correction after
    obs.norv_sequence(RV=obs.planet.RV_sys.value[0])  # offset the RVs so that they are 0 at mid-transit
    
    vshift = -obs.berv0 + obs.RV_sys + obs.vrp + obs.mid_vrp + config_dict['RV_inj']

    # Scale the inputted model
    mod_spec_scaled = (mod_spec - (p.R_pl / p.R_star)**2) * config_dict['scaling_factor']

    if plot_model:
        # Save a plot of the scaled model
        plot_scaled_model(wave_mod, mod_spec, mod_spec_scaled, path_fig)

    # Get list of all exposures
    with open(str(config_dict['obs_dir']) + '/' + 'list_tcorr.txt') as f:
        exp_list = f.readlines()

    # Initialize lists to store data
    wavelengths = []
    counts = []

    for i, exp in enumerate(exp_list):

        # Shift the model wavelength into the observer rest frame for this exposure
        wave_mod_shifted = wave_mod * (1 + vshift[i] / 299792.458)

        # Setup a function to interpolate over the model
        interp_wavelength = interp1d(wave_mod_shifted, mod_spec_scaled, kind='cubic', bounds_error=False, fill_value=0.)
        
        if debug:
            # Plot each exposure
            plt.figure(figsize=(8,3), dpi=200)
        
        # load the exposure
        with fits.open(str(config_dict['obs_dir']) + '/' + exp.strip(), memmap=False) as hdul:

            count = hdul[1].data  # copy of the flux data
            wv = hdul[2].data / 1000
            
            new_count = count.copy()  # create a copy to modify

            # Iterate over the orders
            for w in range(len(wv)):

                if debug:
                    # Plot original 
                    if w == 0: plt.plot(wv[w], count[w], label='Original', zorder=1)
                    else: plt.plot(wv[w], count[w], zorder=1)

                # Interpolate the model to the wavelength, and scale by the lightcurve
                mod_interp = 1 - interp_wavelength(wv[w]) * Wc[i]

                # Multiply the count by the model in that range
                new_count[w] = count[w] * mod_interp

                if debug:
                    # Plot new
                    if w == 0: plt.plot(wv[w], new_count[w], label='Injected', color = 'blue', zorder=0)
                    else: plt.plot(wv[w], new_count[w], color = 'blue', zorder=0)
            
            # Assign the modified data back to the HDU
            hdul[1].data = new_count

            # Save to new fits file with same name, in new folder
            hdul.writeto(str(save_dir / exp.strip()), overwrite=True)

            # Append wavelengths and counts for this exposure to the lists
            wavelengths.append(wv)
            counts.append(hdul[1].data)

            if debug:
                plt.title(f'Exposure {i + 1}')
                plt.legend()
                # plt.show()

    # Copy e2ds files into scratch so we can use it as a new obs_dir
    # Get list of all exposures
    with open(str(config_dict['obs_dir']) + '/' + f'list_e2ds.txt') as f:
        e2ds_list = f.readlines()

    for i, e2ds in enumerate(e2ds_list):
        os.system(f'cp {config_dict["obs_dir"]}/{e2ds.strip()} {save_dir}')

    # Copy the e2ds and tcorr lists into the scratch
    os.system(f'cp {config_dict["obs_dir"]}/list_tcorr.txt {save_dir}')
    os.system(f'cp {config_dict["obs_dir"]}/list_e2ds.txt {save_dir}')


# Lists of data files
list_filenames = {'list_e2ds': 'list_e2ds.txt',
                  'list_tcorr': 'list_tcorr.txt'}

# Mid-transit time (BJD) of each TRAPPIST-1 visit (calculated from Agol et al. 2024)
mid_tr_dict = {
        # SPIRou b
               "2019-06-14": 2458649.07123371738 * u.d,
               "2019-09-25": 2458751.8112203472  * u.d,
               "2020-05-31": 2459001.10676193183 * u.d,
               "2020-08-04": 2459066.07481311538 * u.d,
               "2020-09-08": 2459100.82490738839 * u.d,
               "2020-09-20": 2459112.91190875796 * u.d,
               "2020-09-26": 2459118.95517847751 * u.d,
               "2020-09-29": 2459121.9775049779  * u.d,
               "2021-10-21": 2459508.76251261068 * u.d,
               "2021-10-24": 2459511.78471495466 * u.d,
               "2021-10-27": 2459514.8059938763  * u.d,
        # NIRPS b
               "2023-09-07": 2460194.7007329288  * u.d,
               "2024-07-05": 2460496.8725766301  * u.d,
               "2024-07-11b": 2460502.9159761349 * u.d,
        # NIRPS d
               "2023-06-08": 2460103.8951073092 * u.d,
               "2023-08-12": 2460168.6902942523 * u.d,
               "2023-08-24": 2460180.8397053848 * u.d,
               "2023-10-20": 2460237.5365671548 * u.d,
        # NIRPS e
               "2023-07-17": 2460142.858565431  * u.d,
               "2023-11-04": 2460252.6376593957 * u.d,
               "2024-07-11e": 2460502.7365562134 * u.d,
        # NIRPS f
               "2023-08-28": 2460184.827486578  * u.d,
               "2024-11-02": 2460617.5481359637 * u.d,
               "2025-08-06": 2460893.795434583  * u.d,
        # NIRPS g
               "2023-10-05": 2460222.5638302773 * u.d,
               "2023-11-11": 2460259.6204310769 * u.d,
               "2025-07-08": 2460864.8974261757 * u.d,
                }

# Fixed reduction parameters
cbp = True  # correct bad pixels
iout_all = ['all']
polynome = [False] 
do_tr = [1]
transit_tags = [None]  # use if want to remove spectra

kwargs_gen_tr = {
    'coeffs' : coeffs,
    'ld_model' : ld_model,
    'do_tr' : do_tr,
    'kind_trans' : kind_trans,
    'polynome' : polynome,
    'cbp': cbp }

kwargs_build_ts = {
    'clip_ratio' : 6,
    'clip_ts' : 6,
    'unberv_it' : True }

params_all=[[tellu_frac, mask_wings, 51, 41, 5, nPC, 5.0, 5.0, 5.0, 5.0]]

# Cross-correlation parameters
corrRV0 = np.arange(-corr_xlim, corr_xlim, step_RV_inj)  # x axis (Vrad)

# Model variables
model_filelist = list(model_dir.glob("*.npz"))
model_results = []  # store the model information and cross-correlation results
results_name = f"{models_shortname}_results"



for i, model_filename in enumerate(model_filelist):

    ## INJECTION ##
    
    # Load the model
    model_name = str(model_filename.stem)[22:]
    model_file = np.load(model_filename)
    wave_mod, model_spec, MMW = model_file['wave_mod'], model_file['model_spec'], model_file['MMW']

    # Only run this loop for certain models
    if only_specified:
        if float(model_name.split('_')[2]) not in mole_abundances:
            continue
        if float(model_name.split('_')[4]) not in cloud:
            continue

    print(f"\033[1mModel {i + 1} of {len(model_filelist)}: {model_name}\033[0m")

    # Reduction version
    reduction = f"injected_{model_name}_at{RV_inj}_scaling{scaling}"

    # Data required for the injection
    config_dict = {
                    "instrument" : instrument,
                    "reduction" : reduction,
                    'scaling_factor' : scaling,
                    'RV_inj' : RV_inj,
                  }

    pl_obs.log.setLevel("WARNING")  # do not print all the log information

    for visit_name in nightlist:
        
        # Directory where the data is
        obs_dir = f"/home/mathisb/Github/HRS_data/{instrument}/TRAPPIST-1/{visit_name}/"
        config_dict['obs_dir'] = obs_dir
        
        mid_tr = mid_tr_dict[visit_name]
        
        # Create obs and planet objects
        obs_inj = Observations(name=pl_name, instrument='NIRPS-APERO',
                       pl_kwargs = {'M_star': M_star, 
                                    'R_star': R_star,
                                    'ap': ap,
                                    'R_pl': R_pl,
                                    'mid_tr': mid_tr,
                                    't_peri': mid_tr,
                                    'excent': e,
                                    'w': w,
                                    'RV_sys': RVsys})
    
        obs_inj.fetch_data(obs_dir, **list_filenames, CADC=True)
        p = obs_inj.planet
        
        # Inject
        inj_rec(config_dict, p, obs_inj, visit_name, wave_mod, model_spec)



    ## REDUCTION ##

    obs_dict = {}

    for visit_name in nightlist:

        # Where to output the reductions
        pl_name_fname = ''.join(pl_name.split())
        out_dir = Path(os.environ['SCRATCH']) / f'HRS_reductions/{instrument}/{pl_name_fname}/{visit_name}/{reduction}'
        out_dir.mkdir(parents=True, exist_ok=True)  # make sure the directory exists
        
        # Where to save figures
        path_fig = out_dir / 'Figures/'
        path_fig.mkdir(parents=True, exist_ok=True)  # make sure the  directory exists

        # Check if this reduction has already been done (if so, skip it)
        out_filename1 = f'sequence_{nPC}-pc_mask_wings97'
        out_filename2 = f'retrieval_input_{nPC}-pc_mask_wings97'
        
        out_filename1_full = out_dir / (out_filename1 + f'_data_trs_{visit_name}.npz')

        if out_filename1_full.is_file():
            print(f"Reduction already exists: {visit_name}/{reduction}/{out_filename1}")
            continue

        # Directory with the injected data files
        inj_dir = f"/home/mathisb/scratch/HRS_reductions/{instrument}/{pl_name_fname}/injected_data/{visit_name}/{reduction}"

        mid_tr = mid_tr_dict[visit_name]

        # Create obs object
        obs = Observations(name=pl_name, instrument='NIRPS-APERO',
                       pl_kwargs = {'M_star': M_star, 
                                    'R_star': R_star,
                                    'ap': ap,
                                    'R_pl': R_pl,
                                    'mid_tr': mid_tr,
                                    't_peri': mid_tr,
                                    'excent': e,
                                    'w': w,
                                    'RV_sys': RVsys})
    
        obs.fetch_data(inj_dir, **list_filenames, CADC=True)
        obs_dict[visit_name] = obs

        if visit_name == '2025-07-08':
            transit_tags = [[0,1,2,3,4,5,6,7,8]]
        
        # Do the reduction
        list_tr = pl_obs.generate_all_transits(obs_dict[visit_name], transit_tags, [RVsys.value], params_all, iout_all, **kwargs_gen_tr, **kwargs_build_ts)
        
        # Save sequence with all reduction steps
        pl_obs.save_single_sequences(out_filename1, list_tr['1'], path=out_dir, filename_end=visit_name, save_all=True)

        # Save sequence with only the info needed for a retrieval
        pl_obs.save_sequences(out_filename2, list_tr, do_tr, path=out_dir)

        
        
    ## CROSS-CORRELATION ##

    n_pc_list = []
    mask_wings_list = []
    all_obs = dict()
    all_ccf_map = dict()
    all_logl_map = dict()

    combined_ccf = []
    combined_logl = []
    combined_obs = []
    visit_dict = dict()
    all_visits_str = ""

    for seq_idx, seq_visit in enumerate(nightlist):

        seq = f"sequence_{nPC}-pc_mask_wings97_data_trs_{seq_visit}.npz"
        seq_nb = str(seq_idx + 1)

        # Filenames and paths
        path_reduc = Path(os.environ['SCRATCH']) / f'HRS_reductions/{instrument}/{pl_name_fname}/{seq_visit}/{reduction}'  # where to find the reductions
        path_corr = Path(os.environ['SCRATCH']) / f'HRS_correlations/{instrument}/{pl_name_fname}/{seq_visit}/{reduction}'  # where to output the correlations
        path_corr.mkdir(parents=True, exist_ok=True)  # create output directory if it does not exist
        fig_dir = path_corr / 'Figures'  # where to put the figures
        fig_dir.mkdir(parents=True, exist_ok=True)  # create figures directory if it does not exist

        out_filename = f'{Path(seq).stem}_ccf_logl_seq_{model_name}'

        # Load the observation
        mid_tr = mid_tr_dict[seq_visit]
        pl_kwargs['mid_tr'] = mid_tr
        pl_kwargs['t_peri'] = mid_tr
    
        planet_obj = Planet(pl_name, **pl_kwargs)
        obs = pl_obs.load_single_sequences(seq, pl_name, planet=planet_obj, path=path_reduc, load_all=False, filename_end='', plot=False)
    
        # Generate Kp 
        Kp_array = np.array([obs.Kp.value]) 
    
        n_pc = int(obs.params[5])
        n_pc_list.append(n_pc)
    
        mask_wings_list.append(int(obs.params[1] * 100))

        try:
            # Check if already generated
            saved_values = np.load(path_corr / Path(out_filename + '.npz'))
            print(f"Correlation already exists: {seq_visit}/{reduction}/{out_filename}")
            ccf_map = saved_values['corr']
            logl_map = saved_values['logl']

        except FileNotFoundError:
            # Generate 1d correlations
            ccf_map, logl_map = corr.calc_logl_injred(obs, 'seq', p, Kp_array, corrRV0,
                                                      [n_pc], wave_mod, model_spec, kind_trans)
    
            # Save the correlation in a file
            corr.save_logl_seq(path_corr / Path(out_filename), ccf_map, logl_map,
                               wave_mod, model_spec, n_pc, Kp_array, corrRV0, kind_trans)

        all_obs[(seq_visit, n_pc, mask_wings)] = obs
        all_ccf_map[(seq_visit, n_pc, mask_wings)] = ccf_map
        all_logl_map[(seq_visit, n_pc, mask_wings)] = logl_map

        all_visits_str += seq_nb + "-"
        visit_dict[seq_nb] = obs
        combined_obs.append(obs)
        combined_ccf.append(ccf_map)

        args = [all_something[(seq_visit, nPC, mask_wings)] for all_something in [all_obs, all_ccf_map, all_logl_map]]

        order_indices = pf.select_orders_plot(args[0], wave_mod, model_spec, orders_to_delete=orders_to_delete.get(seq_visit),
                                              max_masked_ratio=max_masked_ratio, min_SNR=min_SNR, min_model_ppm=min_model_ppm, print_orders=False)

        if plot_all_orders:
            fig_name = f'sequence_{nPC}-pc_mask_wings97_data_trs_{seq_visit}_{reduction}_ccf_logl_seq_{model_name}'
            ccf_obj, logl_obj = cc.plot_ccflogl(*args, corrRV0, Kp_array, [nPC], orders=order_indices,
                                            cmap="viridis", path_fig=str(fig_dir) + "/", fig_name=fig_name)
            plt.show()
        else:
            ccf_obj = Correlations(ccf_map, kind="logl", rv_grid=corrRV0, n_pcas=nPC, kp_array=Kp_array)
            ccf_obj.calc_ccf2d(obs, ccf=None, kind='logl_corr', id_pc=None, remove_mean=False, index=None, orders=order_indices)

        if nPC == 0:
            # Mask strong negative correlation regions
            ccf_sum = np.sum(ccf_obj.map_prf, axis=0)      # sum of CCFs over exposures
            noise_mask = ((np.abs(corrRV0) > 10) & (np.abs(corrRV0 - RV_inj) > 10))  # exclude RV region where planet and injected signals are located
    
            # Calculate MAD of CCF sum (like a std dev but without outliers)
            ccf_sum_med = np.median(ccf_sum[noise_mask])
            mad = np.median(np.abs(ccf_sum[noise_mask] - ccf_sum_med))
    
            ccf_obj.map_prf[:, ccf_sum < ccf_sum_med - 4 * mad] = np.ma.masked   # mask negative outlier regions
     
        if len(nightlist) == 1:
            if plot_KpVrad:
                KpVrad_map = pf.calculate_KpVsys_map(obs, ccf_obj, method='clip', rv_grid=corrRV0, kp_grid=kp_grid,
                                            clip_sigma=clip_sigma, clip_iter=clip_iter, output_map=True, expected_vrad=RV_inj, title=False, savefig=None)
    
            if plot_all_orders:
                pf.plot_all_orders_correl(corrRV0, ccf_obj.data.squeeze(), obs, icorr=None, logl=False, sharey=True,
                                  vrp=np.zeros_like(obs.vrp), RV_sys=0.0, vmin=None, vmax=None, common_scale=common_scale, vline=None,
                                  hline=2, kind='snr', return_snr=False, orders=order_indices)


    
    ## COMBINED ##

    if len(nightlist) > 1:
        transit_tags = [np.arange(obs.n_spec) for obs in visit_dict.values()]
        all_visits = pl_obs.gen_merge_obs_sequence(obs, visit_dict, np.arange(1, len(nightlist) + 1), None, coeffs, ld_model, kind_trans, light=True)
        all_visits.dt = np.concatenate([vst.dt for vst in visit_dict.values()])  # exposure times
        all_visits_str = all_visits_str[:-1]
        visit_dict[all_visits_str.replace("-", "")] = all_visits

        # Where to output the combined correlations
        output_dir_comb = Path(os.environ['SCRATCH']) / f'HRS_correlations/{instrument}/{pl_name_fname}/Combined/{reduction}'
        
        # Figures directory
        fig_dir_comb = str(output_dir_comb) + "/Figures/"
        Path(fig_dir_comb).mkdir(parents=True, exist_ok=True)
        
        # Create filename for combined correlation file
        args_filename = []

        for i in range(len(nightlist)):
            args_filename.append((str(visit_dict[str(i + 1)].params[5]), str(int(visit_dict[str(i + 1)].params[1] * 100))))
        
        args_filename = ['-'.join(values) for values in zip(*args_filename)]
        filename = f"combined_corr_nights{all_visits_str}" + "_pc{}_mask_wings{}_".format(*args_filename) + f"{model_name}"

        # Get list of obs objects
        obs_list = list(visit_dict.values())[:-1]
        
        # Choose what orders to plot
        comb_orders_to_delete = pf.combined_orders_to_delete(orders_to_delete, nightlist)
        idx_orders = pf.select_orders_plot(obs_list, wave_mod, model_spec, orders_to_delete=comb_orders_to_delete,
                                              max_masked_ratio=max_masked_ratio, min_SNR=min_SNR, min_model_ppm=min_model_ppm)
        
        # Generate CCF
        ccf_map_comb = np.concatenate(combined_ccf)

        ccf_obj_comb = Correlations(ccf_map_comb, kind="logl", rv_grid=corrRV0, n_pcas=nPC, kp_array=Kp_array)
        ccf_obj_comb.calc_ccf2d(all_visits, ccf=None, kind='logl_corr', id_pc=None, remove_mean=False, index=None, orders=idx_orders)

        if nPC == 0:
            # Mask strong negative correlation regions
            ccf_sum_mask_list = []
            
            for c in combined_ccf:
                ccf_sum = np.sum(np.sum(c.squeeze()[:, idx_orders], axis=1), axis=0)  # sum of CCFs over orders and exposures
                noise_mask = ((np.abs(corrRV0) > 10) & (np.abs(corrRV0 - RV_inj) > 10))  # exclude RV region where planet and injected signals are located
            
                # Calculate MAD of CCF sum (like a std dev but without outliers)
                ccf_sum_med = np.median(ccf_sum[noise_mask])
                mad = np.median(np.abs(ccf_sum[noise_mask] - ccf_sum_med))
            
                ccf_sum_mask_list.append(ccf_sum < ccf_sum_med - 4 * mad)  # mask negative outlier regions
                
            combined_ccf_sum_mask = np.any(ccf_sum_mask_list, axis=0)
            ccf_obj_comb.map_prf[:, combined_ccf_sum_mask] = np.ma.masked

        # Save combined correlation
        pl_obs.save_sequences(filename, visit_dict, [int(all_visits_str.replace("-", ""))], path=output_dir_comb, print_out=False)

        if plot_KpVrad:
            KpVrad_map = pf.calculate_KpVsys_map(all_visits, ccf_obj_comb, method='clip', rv_grid=corrRV0, kp_grid=kp_grid,
                                                clip_sigma=clip_sigma, clip_iter=clip_iter, output_map=True, expected_vrad=RV_inj, title=False, savefig=None)


   
    ## SAVE CORRELATION MAP ##        

    if plot_KpVrad:
        # Store the results of each model
        model_results.append({
            "Nights": nightlist, 
            "Scenario": scenario,
            gg: float(model_name.split('_')[2]),
            "Cloud": float(model_name.split('_')[4]),
            "MMW": float(MMW),
            "RV_injection": RV_inj,
            "Planet_Kp": obs.Kp.value,
            "KpVrad_map": KpVrad_map,
        })
    
    print('===\n')
    pl_obs.log.setLevel("INFO")  # go back to normal log printing



print("=== ALL INJECTIONS, REDUCTIONS AND CROSS-CORRELATIONS DONE ===")

if plot_KpVrad:
    if len(nightlist) > 1:
        results_name += f'_{all_visits_str}.npy'
    else:
        results_name += f'_{nightlist[0]}.npy'

    results_dir = Path('/home/mathisb/Github/HRS_models/' + models_shortname)
    results_dir.mkdir(parents=True, exist_ok=True)
        
    np.save(results_dir / results_name, model_results, allow_pickle=True)
    print("=== RESULTS SAVED IN FILE ===")




    
# TODO
# add N2rich ?