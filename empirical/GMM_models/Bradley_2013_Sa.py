"""
Bradley_2010_Sa.py
Richard Clare 
16/3/16


Translated from Bradley_2010_Sa.m  (Brendon Bradley   4 June 2010)

Provides the attenuation relation for Sa in units of g (also PGV).

This model is that developed by Bradley (2010) for prediction of Sa and
PGV in NZ crustal tectonic region.  It is based on the Chiou and Youngs
2008 NGA relation.  Modifications include: (i) the use of the results of Chiou
et al 2010 small-moderate magnitude model where they reduce the amplitude
of small magnitude events for short periods; (ii) incorporation of a 
volcanic anelatic attenuation term, (iii) extension of site effect for 
site class A, and (iv) adjusted normal event scaling for short periods; 
(v) removed consideration of aftershocks.

reference: Chiou, B., Youngs, R. R., Abrahamson, N. A., Addo, K., 2010. 
Ground-motion attenuation model for small-to-moderate shallow crustal 
earthquakes in california and Its implications on regionalization of 
ground-motion prediction models, Earthquake Spectra,  (to appear).

The reference above provides only the modified coefficients for PGA, PGV,
Sa(0.3) and Sa(1.0).  The coefficeints for other periods have been
obtained from interpolation (check NZ applicability paper, to appear, for
references).


Input Variables:
 siteprop      = properties of site (soil etc)
                 siteprop.Rrup  = Source-to-site distance (km) (Rrup distance)
                 siteprop.V30   -'(any real variable)' shear wave velocity(m/s)
                 siteprop.V30measured - yes =1 (i.e. from Vs tests); no =
                   0 (i.e. estimated from geology)
                 siteprop.Rrup -'closest distance coseismic rupture (km)
                 siteprop.Rx -distance measured perpendicular to fault
                   strike from surface projection of updip edge of the
                   fault rupture (+ve in downdip dir) (km)
                 siteprop.Rtvz - source-to-site distance in the Taupo
                                 volcanic zone (TVZ) in km.
                 siteprop.period -'(-1),(0),(real variable)' period of vibration =-1->PGV; =0->PGA; >0->SA
                 siteprop.Z1pt0 -'depth to the 1.0km/s shear wave velocity horizon (optional, uses default relationship otherwise)

 faultprop     = properties of fault (strikeslip etc)
                 faultprop.Mw= Moment magnitude (Mw)
                 faultprop.Ztor -'depth to top of coseismic rupture (km)
                 faultprop.rake -'rake angle in degrees
                 faultprop.dip -'avg dip angle in degrees

Output Variables:
 SA           = median SA  (or PGA or PGV)
 sigma_SA     = lognormal standard deviation of SA
                %sigma_SA(1) = total std
                 %sigma_SA(2) = interevent std
                 %sigma_SA(3) = intraevent std

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Coefficients
 	// coefficients (index -1 is PPV and 0 is PGA): 

Issues:

"""
import numpy as np
from matplotlib.mlab import find
from empirical.util.classdef import interpolate_to_closest

period_list = np.array([-1, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5,
               7.5, 10])

c1 = [2.3132, -1.1985, -1.1958, -1.1756, -1.0909, -0.9793, -0.8549, -0.6008, -0.4700, -0.4139, -0.5237, -0.6678,
      -0.8277, -1.1284, -1.3926, -1.8664, -2.1935, -2.6883, -3.1040, -3.7085, -4.1486, -4.4881, -5.0891, -5.5530]

c1a = [0.1094, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0999, 0.0997, 0.0991, 0.0936, 0.0766,
       0.0022, -0.0591, -0.0931, -0.0982, -0.0994, -0.0999, -0.1]

# Modification 1: Normal style of faulting at short periods###########
c1b = [-0.0626, -0.455, -0.455, -0.455, -0.455, -0.455, -0.455, -0.454, -0.453, -0.45, -0.4149, -0.3582, -0.3113,
       -0.2646, -0.2272, -0.162, -0.14, -0.1184, -0.11, -0.104, -0.102, -0.101, -0.101, -0.1]

c2 = 1.06
# Modification 1: Scaling at small Mag
c3 = [2.29445, 1.50000, 1.50299, 1.50845, 1.51549, 1.52380, 1.53319, 1.56053, 1.59241, 1.66640, 1.75021, 1.84052,
      1.93480, 2.12764, 2.31684, 2.73064, 3.03000, 3.43384, 3.67464, 3.64933, 3.60999, 3.50000, 3.45000, 3.45000]

cm = [5.49000, 5.85000, 5.81711, 5.80023, 5.78659, 5.77472, 5.76402, 5.74056, 5.72017, 5.68493, 5.65435, 5.62686,
      5.60162, 5.55602, 5.51513, 5.38632, 5.31000, 5.29995, 5.32730, 5.43850, 5.59770, 5.72760, 5.98910, 6.19300]

cn = [1.648, 2.996, 2.996, 3.292, 3.514, 3.563, 3.547, 3.448, 3.312, 3.044, 2.831, 2.658, 2.505, 2.261, 2.087,
      1.812, 1.648, 1.511, 1.47, 1.456, 1.465, 1.478, 1.498, 1.502]

c4 = -2.1
c4a = -0.5
crb = 50.0

c5 = [5.17, 6.16, 6.16, 6.158, 6.155, 6.1508, 6.1441, 6.12, 6.085, 5.9871, 5.8699, 5.7547, 5.6527, 5.4997, 5.4029,
      5.29, 5.248, 5.2194, 5.2099, 5.204, 5.202, 5.201, 5.2, 5.2]

c6 = [0.4407, 0.4893, 0.4893, 0.4892, 0.489, 0.4888, 0.4884, 0.4872, 0.4854, 0.4808, 0.4755, 0.4706, 0.4665, 0.4607,
      0.4571, 0.4531, 0.4517, 0.4507, 0.4504, 0.4501, 0.4501, 0.45, 0.45, 0.45]

chm = 3.0

c7 = [0.0207, 0.0512, 0.0512, 0.0512, 0.0511, 0.0508, 0.0504, 0.0495, 0.0489, 0.0479, 0.0471, 0.0464, 0.0458,
      0.0445, 0.0429, 0.0387, 0.035, 0.028, 0.0213, 0.0106, 0.0041, 0.001, 0, 0]

# Modification: Ztor maximum, c8
c8 = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10.5, 11, 12, 13, 14, 15, 16, 18, 19, 19.75, 20, 20, 20]

c9 = [0.3079, 0.79, 0.79, 0.8129, 0.8439, 0.874, 0.8996, 0.9442, 0.9677, 0.966, 0.9334, 0.8946, 0.859, 0.8019,
      0.7578, 0.6788, 0.6196, 0.5101, 0.3917, 0.1244, 0.0086, 0, 0, 0]

c9a = [2.669, 1.5005, 1.5005, 1.5028, 1.5071, 1.5138, 1.523, 1.5597, 1.6104, 1.7549, 1.9157, 2.0709, 2.2005, 2.3886,
       2.5, 2.6224, 2.669, 2.6985, 2.7085, 2.7145, 2.7164, 2.7172, 2.7177, 2.718]

# modification: change of anelastic attenuation
cy1 = [-0.0033, -0.0096, -0.0096, -0.0097, -0.0101, -0.0105, -0.0109, -0.0117, -0.0117, -0.0111, -0.0100, -0.0091,
       -0.0082, -0.0069, -0.0059, -0.0045, -0.0037, -0.0028, -0.0023, -0.0019, -0.0018, -0.0017, -0.0017, -0.0017]

cy2 = [-0.00687, -0.00480, -0.00481, -0.00486, -0.00503, -0.00526, -0.00549, -0.00588, -0.00591, -0.00540, -0.00479,
       -0.00427, -0.00384, -0.00317, -0.00272, -0.00209, -0.00175, -0.00142, -0.00143, -0.00115, -0.00104, -0.00099,
       -0.00094, -0.00091]

cy3 = 4.0

# modification: inclusion of TVZ attenuation
ctvz = [5.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.5000,
        3.2000, 3.5000, 4.500, 5.0000, 5.4000, 5.8000, 6.0000, 6.1500, 6.3000, 6.4250, 6.5500]

phi1 = [-0.7861, -0.4417, -0.4417, -0.434, -0.4177, -0.4, -0.3903, -0.404, -0.4423, -0.5162, -0.5697, -0.6109,
        -0.6444, -0.6931, -0.7246, -0.7708, -0.799, -0.8382, -0.8663, -0.9032, -0.9231, -0.9222, -0.8346, -0.7332]

phi2 = [-0.0699, -0.1417, -0.1417, -0.1364, -0.1403, -0.1591, -0.1862, -0.2538, -0.2943, -0.3113, -0.2927, -0.2662,
        -0.2405, -0.1975, -0.1633, -0.1028, -0.0699, -0.0425, -0.0302, -0.0129, -0.0016, 0, 0, 0]

phi3 = [-0.008444, -0.00701, -0.00701, -0.007279, -0.007354, -0.006977, -0.006467, -0.005734, -0.005604, -0.005845,
        -0.006141, -0.006439, -0.006704, -0.007125, -0.007435, -0.00812, -0.008444, -0.007707, -0.004792, -0.001828,
        -0.001523, -0.00144, -0.001369, -0.001361]

phi4 = [5.41, 0.102151, 0.102151, 0.10836, 0.119888, 0.133641, 0.148927, 0.190596, 0.230662, 0.266468, 0.255253,
        0.231541, 0.207277, 0.165464, 0.133828, 0.085153, 0.058595, 0.031787, 0.019716, 0.009643, 0.005379,
        0.003223, 0.001134, 0.000515]

phi5 = [0.2899, 0.2289, 0.2289, 0.2289, 0.2289, 0.2289, 0.229, 0.2292, 0.2297, 0.2326, 0.2386, 0.2497, 0.2674,
        0.312, 0.361, 0.4353, 0.4629, 0.4756, 0.4785, 0.4796, 0.4799, 0.4799, 0.48, 0.48]

phi6 = [0.006718, 0.014996, 0.014996, 0.014996, 0.014996, 0.014996, 0.014996, 0.014996, 0.014996, 0.014988,
        0.014964, 0.014881, 0.014639, 0.013493, 0.011133, 0.006739, 0.005749, 0.005544, 0.005521, 0.005517,
        0.005517, 0.005517, 0.005517, 0.005517]

phi7 = [459, 580, 580, 580, 580, 579.9, 579.9, 579.6, 579.2, 577.2, 573.9, 568.5, 560.5, 540, 512.9, 441.9, 391.8,
        348.1, 332.5, 324.1, 321.7, 320.9, 320.3, 320.1]

phi8 = [0.1138, 0.07, 0.07, 0.0699, 0.0701, 0.0702, 0.0701, 0.0686, 0.0646, 0.0494, -0.0019, -0.0479, -0.0756,
        -0.096, -0.0998, -0.0765, -0.0412, 0.014, 0.0544, 0.1232, 0.1859, 0.2295, 0.266, 0.2682]

tau1 = [0.2539, 0.3437, 0.3437, 0.3471, 0.3603, 0.3718, 0.3848, 0.3878, 0.3835, 0.3719, 0.3601, 0.3522, 0.3438,
        0.3351, 0.3353, 0.3429, 0.3577, 0.3769, 0.4023, 0.4406, 0.4784, 0.5074, 0.5328, 0.5542]

tau2 = [0.2381, 0.2637, 0.2637, 0.2671, 0.2803, 0.2918, 0.3048, 0.3129, 0.3152, 0.3128, 0.3076, 0.3047, 0.3005,
        0.2984, 0.3036, 0.3205, 0.3419, 0.3703, 0.4023, 0.4406, 0.4784, 0.5074, 0.5328, 0.5542]

sigma1 = [0.4496, 0.4458, 0.4458, 0.4458, 0.4535, 0.4589, 0.463, 0.4702, 0.4747, 0.4798, 0.4816, 0.4815, 0.4801,
          0.4758, 0.471, 0.4621, 0.4581, 0.4493, 0.4459, 0.4433, 0.4424, 0.442, 0.4416, 0.4414]

sigma2 = [0.3554, 0.3459, 0.3459, 0.3459, 0.3537, 0.3592, 0.3635, 0.3713, 0.3769, 0.3847, 0.3902, 0.3946, 0.3981,
          0.4036, 0.4079, 0.4157, 0.4213, 0.4213, 0.4213, 0.4213, 0.4213, 0.4213, 0.4213, 0.4213]

sigma3 = [0.7504, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7999, 0.7997, 0.7988, 0.7966, 0.7792, 0.7504,
          0.7136, 0.7035, 0.7006, 0.7001, 0.7, 0.7, 0.7]


def Bradley_2013_Sa(siteprop, faultprop, im, periods=None):
    # declare a whole bunch of coefficients
    ##################

    if im == 'PGA':
        periods = [0]
    if im == 'PGV':
        periods = [-1]

    results = []

    for period in periods:

        T = period

        tol = 0.0001  # tolerance to the recorded period values before we interpolate

        closestIndex = int(np.argmin(np.abs(period_list - T)))
        closestPeriod = period_list[closestIndex]
        if not np.isclose(closestPeriod, T):  # interpolate between periods if neccesary

            # find the period values above and below
            T_low = period_list[T >= period_list][-1]
            T_high = period_list[T <= period_list][0]
            
            # recursively call this function for the periods above and below
            brad_low = calculate_Bradley(siteprop, faultprop, T_low)
            brad_high = calculate_Bradley(siteprop, faultprop, T_high)
            sigma_SA = []  # initialize empty list
      
            # now interpolate the low and high values
            result = interpolate_to_closest(T, T_high, T_low, brad_high, brad_low)

        else:
            result = calculate_Bradley(siteprop, faultprop, T)
        results.append(result)

    if im in ['PGA', 'PGV']:
        results = results[0]

    return results


def calculate_Bradley(siteprop, faultprop, period):
    M = faultprop.Mw
    Rrup = siteprop.Rrup
    Rjb = siteprop.Rjb
    Rx = siteprop.Rx
    Vs30 = siteprop.vs30
    if siteprop.z1p0 < 0:  # depth to 1.0 km/s Vs horizon
        Z10 = np.exp(28.5 - 3.82 / 8 * np.log(Vs30 ** 8. + 378.7 ** 8.))
    else:
        Z10 = siteprop.z1p0 * 1000  # Convert from km to m
    delta = faultprop.dip  # dip in degrees
    Lambda = faultprop.rake  # rake in degrees			#lambda is a keyword in Python so changed to Lambda
    Ztor = faultprop.ztor
    Rtvz = siteprop.Rtvz
    deltar = delta * np.pi / 180.0
    frv = (Lambda >= 30) & (Lambda <= 150)  # frv: 1 for lambda between 30 and 150, 0 otherwise
    fnm = (Lambda >= -120) & (Lambda <= -60)  # fnm: 1 for lambda between -120 and -60, 0 otherwise
    HW = Rx >= 0

    closest_index = int(np.argmin(np.abs(period_list - period)))

    f_inferred = None
    f_measured = None
    if siteprop.vs30measured == 1:
        f_inferred = 0  # 1: Vs30 is measured.
        f_measured = 1
    else:
        f_inferred = 1  # 1: Vs30 inferred.
        f_measured = 0

    # don't interpolate
    i = closest_index

    # modifications from CY10 (i.e. CY08 also)
    # 1) maximum Ztor set to 10km depth
    # 2) insertion of v1 term which can be used to obtain site effect for up
    # to Vs30 = 1500 m/s
    # 3) Something for volcanic anelastic attenuation
    # 4) Changed normal faulting style effect for short periods

    # calculate terms in the median computation
    term1 = c1[i]
    # Modification 2: Ztor maximum depth
    term2 = (c1a[i] * frv + c1b[i] * fnm + c7[i] * (np.min((Ztor, c8[i])) - 4.))  # modification of Ztor limit
    term5 = c2 * (M - 6.)

    term6 = ((c2 - c3[i]) / cn[i]) * np.log(1. + np.exp(cn[i] * (cm[i] - M)))

    term7 = c4 * np.log(Rrup + c5[i] * np.cosh(c6[i] * np.max((M - chm, 0))))

    term8 = (c4a - c4) * np.log(np.sqrt(Rrup ** 2 + (crb) ** 2))

    # Modification 3: Rtvz attenuation
    term9 = (cy1[i] + cy2[i] / np.cosh(np.max((M - cy3, 0)))) * (
            1. + ctvz[i] * Rtvz / Rrup) * Rrup  # modified term including Rtvz anelastic attenuation

    term10 = c9[i] * HW * np.tanh(Rx * np.cos(deltar) ** 2 / c9a[i]) * (
            1. - np.sqrt(Rjb ** 2 + Ztor ** 2) / (Rrup + 0.001))

    # reference Sa on rock (Vs=1130m/s)
    Sa1130 = np.exp(term1 + term2 + term5 + term6 + term7 + term8 + term9 + term10)

    # Modification 4: Rock amplification
    if period == 0:
        v1 = 1800
    elif period == -1:
        v1 = np.min((np.max((1130. * (1. / 0.75) ** (-0.11), 1130.)), 1800.))
    else:
        v1 = np.min((np.max((1130. * (period / 0.75) ** (-0.11), 1130.)), 1800.))

    term11 = phi1[i] * np.log(np.min((Vs30, v1)) / 1130.)  # modified site term accounting for Vs=1800m/s

    term12 = phi2[i] * (
            np.exp(phi3[i] * (np.min((Vs30, 1130.)) - 360.)) - np.exp(phi3[i] * (1130. - 360.))) * np.log(
        (Sa1130 + phi4[i]) / phi4[i])

    term13 = phi5[i] * (1. - 1. / np.cosh(phi6[i] * np.max((0, Z10 - phi7[i])))) + phi8[i] / np.cosh(
        0.15 * np.max((0, Z10 - 15)))
    
    # Compute median
    Sa = np.exp(np.log(Sa1130) + term11 + term12 + term13)

    # Compute standard deviation
    sigma_SA = compute_stdev(f_inferred, f_measured, M, Sa1130, Vs30, i)
    return Sa, sigma_SA


def compute_stdev(Finferred, Fmeasured, M, Sa1130, Vs30, i):
    b = phi2[i] * (np.exp(phi3[i] * (np.min((Vs30, 1130.)) - 360.)) - np.exp(phi3[i] * (1130. - 360.)))
    c = phi4[i]
    NL0 = b * Sa1130 / (Sa1130 + c)
    sigma = ((sigma1[i] + (sigma2[i] - sigma1[i]) / 2. * (np.min((np.max((M, 5.)), 7.)) - 5.)) *
             np.sqrt(sigma3[i] * Finferred + 0.7 * Fmeasured + (1. + NL0) ** 2))
    tau = tau1[i] + (tau2[i] - tau1[i]) / 2. * (np.min((np.max((M, 5.)), 7.)) - 5.)
    # outputs
    sigma_total = np.sqrt((1. + NL0) ** 2 * tau ** 2 + sigma ** 2)  # 0
    sigma_inter = (1. + NL0) * tau  # 1
    sigma_intra = sigma  # 2
    sigma_SA = [sigma_total, sigma_inter, sigma_intra]

    return sigma_SA
