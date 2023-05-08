# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import ROM
import lal
import lalsimulation as ls
from multiprocessing import Pool
from itertools import combinations 

def frequency(q, mass_total, lambda1, chi_1x=0, chi_1y=0, chi_1z=0,
                chi_2x=0, chi_2y=0, chi_2z=0,
                distance=1,
                inclination=0,
                phase=0.0,
                longAscNodes=0,
                meanPerAno=0,
                eccentricity=0,
                delta_f=1/64,
                f_min=20,
                f_max=4096,
                f_ref=20,
                laldict=lal.CreateDict(),
                binary = ls.IMRPhenomD_NRTidalv2):

    hf_plus_waves = []
    hf_cross_waves = []
    for i in q:
        for j in lambda1:
            ls.SimInspiralWaveformParamsInsertTidalLambda1(laldict, j)  # insert the tidal deformability of star 1
            m_1 = (mass_total*i)/(i+1)
            m_2 = mass_total/(i+1)

            # Generate h+ and hx
            hf_plus_aligned, hf_cross_aligned = ls.SimInspiralChooseFDWaveform(
                m_1, m_2,
                chi_1x, chi_1y, chi_1z,
                chi_2x, chi_2y, chi_2z,
                distance,
                inclination,
                phase,
                longAscNodes,
                meanPerAno,
                eccentricity,
                delta_f,
                f_min,
                f_max,
                f_ref,
                laldict,
                binary
            )

            freqs_aligned  = np.arange(0, len(hf_plus_aligned.data.data))*delta_f + delta_f

            # Truncate waveforms
            index_min = len(freqs_aligned) - len(freqs_aligned[freqs_aligned>f_min])
            index_max = len(freqs_aligned[freqs_aligned<f_max])

            freq_trunc = freqs_aligned[index_min:index_max]
            hf_plus_trunc = hf_plus_aligned.data.data[index_min:index_max]
            hf_cross_trunc = hf_cross_aligned.data.data[index_min:index_max]

            # Geometric unit conversion
            freq_geom = freq_trunc*(mass_total/lal.MSUN_SI)*lal.MTSUN_SI

            hf_plus_waves.append(hf_plus_trunc)
            hf_cross_waves.append(hf_cross_trunc)
        
    return np.asarray(hf_plus_waves), np.asarray(hf_cross_waves), np.asarray(freq_geom)

if __name__ == '__main__':
    main()