###
# This script is the first step in comparing the CMB lensing prediction of axionHMcode with CAMB or cosmopower
# It creates a array storing the transfer function between linear and nonlinear spectra
# above a certain redshift, it assumes no nonlinearities (T=1)
###

import numpy as np
import camb
from camb import model, initialpower

from scipy.interpolate import CubicSpline
import os

import sys
sys.path.append('/home/alex/TD_SV_axionHMcode/axionHMcode/')
sys.path.append('/home/alex/TD_SV_axionHMcode/axionHMcode/axionCAMB_and_lin_PS/')
sys.path.append('/home/alex/TD_SV_axionHMcode/axionHMcode/cosmology/')
sys.path.append('/home/alex/TD_SV_axionHMcode/axionHMcode/axion_functions/')
sys.path.append('/home/alex/TD_SV_axionHMcode/axionHMcode/halo_model/')


from axionCAMB_and_lin_PS import axionCAMB_wrapper 
from axionCAMB_and_lin_PS import load_cosmology  
from axionCAMB_and_lin_PS import lin_power_spectrum 

from halo_model import HMcode_params
from halo_model import PS_nonlin_cold
from halo_model import PS_nonlin_axion

from axion_functions import axion_params
from cosmology.overdensities import func_D_z_unnorm_int

print("############# ", axionCAMB_wrapper.__file__)

def camb_nl_power_spectrum(k_array, cosmo_dic):
    """
    Inputs:
    k_array (array of wavenumbers at which the power spectrum is evaluated)
    cosmo_dic (dictionary of cosmological paramters as used by axionHMcode)
    example input: power_spec_dic_ax['k'], cosmos 
    Output:
    nonlinear spectrum

    DEPRECATED!
    """
    M_arr = np.logspace(cosmo_dic['M_min'], cosmo_dic['M_max'], 100)
    kmax = cosmo_dic['transfer_kmax']

    k_array = k_array[k_array>=5e-4]

    ## CAMB PART ##
    pars =  camb.set_params(H0=100*cosmo_dic['h'], ombh2=cosmo_dic['omega_b_0'],
                        omch2=cosmo_dic['omega_d_0'], ns=cosmo_dic['ns'],
                            mnu=0.06, omk=0, num_massive_neutrinos = 1, nnu=3.044,
                       As=cosmo_dic['As'], halofit_version='mead2016', lmax=3000, tau = 0.06)

    pars.set_matter_power(redshifts=[cosmo_dic['z']], kmax=1.5*kmax, accurate_massive_neutrino_transfers=True,k_per_logint=10)
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=5e-4, maxkh=kmax, npoints = 400)

    pars.NonLinear = model.NonLinear_both
    pars.NonLinearModel.set_params('mead2016', HMCode_A_baryon = 3.13, HMCode_eta_baryon = 0.603)
    results = camb.get_results(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=5e-4, maxkh=kmax, npoints = 400)

    ## AXION HMCODE PART ##
    # no need to rerun axioncamb
    # use lin pk form camb instead
    pk_lin_lcdm = CubicSpline(kh, pk[0])(k_array)

    power_spec_dic = {}
    power_spec_dic['k'] = k_array
    power_spec_dic['power_total'] = pk_lin_lcdm
    power_spec_dic['power_CDM'] = pk_lin_lcdm
    power_spec_dic['power_baryon'] = pk_lin_lcdm
    power_spec_dic['power_cold'] = pk_lin_lcdm
    power_spec_dic['power_axion'] = pk_lin_lcdm

    hmcode_params = HMcode_params.HMCode_param_dic(cosmos, power_spec_dic['k'], power_spec_dic['power_cold'])

    axion_param = axion_params.func_axion_param_dic(M_arr, cosmos, power_spec_dic, hmcode_params, concentration_param=True)

    PS_matter_nonlin = PS_nonlin_axion.func_full_halo_model_ax(M_arr, power_spec_dic,
                                                                      cosmos, hmcode_params, axion_param,
                                                                   alpha = True,
                                                              eta_given = True,
                                                              one_halo_damping = True,
                                                                   two_halo_damping = True, concentration_param=True, full_2h=False)
    pk_axionhmcode_lcdm = PS_matter_nonlin[0]
    interpolated_ratio = CubicSpline(kh_nonlin, pk_nonlin[0])(k_array) / pk_axionhmcode_lcdm

    return interpolated_ratio
    
def neutrino_nonlin_correction(k_array, cosmo_dic):
    """
    Compute the neutrino correction to account for the fact that axionHMcode assumes m_nu=0
    """
    
    kmax = cosmo_dic['transfer_kmax']


    ## NO NEUTRINOS ##
    pars =  camb.set_params(H0=100*cosmo_dic['h'], ombh2=cosmo_dic['omega_b_0'], 
                        omch2=cosmo_dic['omega_d_0'], ns=cosmo_dic['ns'], 
                        mnu=0.0, omk=0,
                       As=cosmo_dic['As'], halofit_version='mead2020', lmax=3000, tau = 0.06)

    
    pars.set_matter_power(redshifts=[cosmo_dic['z']], kmax=kmax)

    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk_nonu = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints = 400)


    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin_nonu = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints = 400)

    
    ## WITH NEUTRINOS ##
    pars =  camb.set_params(H0=100*cosmo_dic['h'], ombh2=cosmo_dic['omega_b_0'],
                        omch2=cosmo_dic['omega_d_0'], ns=cosmo_dic['ns'],
                        mnu=0.06, omk=0,
                       As=cosmo_dic['As'], halofit_version='mead2016', lmax=3000, tau = 0.06)

    pars.set_matter_power(redshifts=[cosmo_dic['z']], kmax=kmax)

    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints = 400)


    pars.NonLinear = model.NonLinear_both
    pars.NonLinearModel.set_params('mead2016', HMCode_A_baryon = 3.13, HMCode_eta_baryon = 0.603)
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints = 400)

    ## CORRECTION ##
    C = (pk_nonlin[0] / pk_nonlin_nonu[0]) / (pk[0] / pk_nonu[0])

    interpolated_C = CubicSpline(kh, C)(k_array)
    
    return interpolated_C



zs = np.loadtxt('/home/alex/axionNL_new/optimal_z_array.dat')

input_file_path = 'input_file.txt'
    
#IMPORTANT:Change here the path to the axionCAMB executable path directory (second path in the function)
# assumes that thee axionCAMB executable is names .camb
axionCAMB_exe_path = '/home/keir/Software/axionHMcode_validation/' #'/home/alex/TD_SV_axionHMcode/axionHMcode/'
    
################################################################################    
# save cosmological parameter in a dictionary 
################################################################################
cosmos = load_cosmology.load_cosmology_input(input_file_path) 

def get_nl_ratio(redshift):
    """
    Main function to compute the nonlinear transfer function: PkNL/PkL
    """

    
    cosmos['z'] = redshift
    cosmos['G_a'] = func_D_z_unnorm_int(cosmos['z'], cosmos['Omega_m_0'], cosmos['Omega_w_0']) # recompute
    
    if redshift <=4:
        axionCAMB_wrapper.axioncamb_params('paramfile_axionCAMB.txt', 
                                           cosmos, massless_neutrinos=2.044, massive_neutrinos=1, omnuh2=0.06/93.14,
                                       transfer_high_precision='T',  transfer_k_per_logint=10, 
                                       accuracy_boost=2, l_accuracy_boost=1, l_sample_boost=1,
                                           output_root=axionCAMB_exe_path + 'cosmos', print_info = True)
        axionCAMB_wrapper.run_axioncamb('paramfile_axionCAMB.txt', 
                                    axionCAMB_exe_path, 
                                        cosmos, print_info = True)



        ################################################################################
        # Create linear power spectra from axionCAMB tranfer functions 
        ################################################################################
        #lin PS on given k range
        power_spec_dic_ax = lin_power_spectrum.func_power_spec_dic(axionCAMB_exe_path + 'cosmos_transfer_out.dat', cosmos)



        ################################################################################
        # Compute parameter related to axions and HMCode2020
        ################################################################################

        M_arr = np.logspace(cosmos['M_min'], cosmos['M_max'], 100)

        
        hmcode_params = HMcode_params.HMCode_param_dic(cosmos, power_spec_dic_ax['k'], power_spec_dic_ax['power_cold'])

        axion_param = axion_params.func_axion_param_dic(M_arr, cosmos, power_spec_dic_ax, hmcode_params, concentration_param=True)

        ################################################################################
        # Caluclate non-linear power spectrum in mixed DM and LCDM cosmology
        ################################################################################
    
        PS_matter_nonlin = PS_nonlin_axion.func_full_halo_model_ax(M_arr, power_spec_dic_ax, 
                                                                      cosmos, hmcode_params, axion_param,
                                                                   alpha = True, 
                                                              eta_given = True, 
                                                              one_halo_damping = True, 
                                                                   two_halo_damping = True, concentration_param=True, full_2h=False)


    if redshift <= 4: 
        nu_nl_corr = neutrino_nonlin_correction(power_spec_dic_ax['k'], cosmos)
        ratio = PS_matter_nonlin[0] / power_spec_dic_ax['power_total']
        ratio = ratio * nu_nl_corr
        pk = PS_matter_nonlin[0] * nu_nl_corr
    else:
        ratio = 1.
        pk = 1.

    return ratio, pk


## RUN FUNCTION AND SAVE OUTPUT ##
ratios = []
pks = []
for z in zs:
    ratio, pk = get_nl_ratio(z)
    ratios.append(ratio)
    pks.append(pk)
    print(z, " done!")


ratios_copy = ratios.copy()
pks_copy = pks.copy()
for xi in range(len(ratios)):
    if np.all(ratios[xi] == 1.):
        ratios_copy[xi] = np.ones(len(ratios[0]))
        pks_copy[xi] = np.ones(len(ratios[0]))
        
np.savetxt('nonlinear_ratios.dat', np.array(ratios_copy))
np.savetxt('nonlinear_pks.dat', np.array(pks_copy))
