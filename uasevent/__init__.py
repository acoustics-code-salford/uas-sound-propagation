'''
Acoustic propagation model for auralising UAS sources in motion
===============================================================

uasevent is a Python programme simulating simple atmospheric acoustic
propagation effects for creating multichannel auralisations of 
unmanned aerial systems (UAS - drones), or other airborne sources,
in motion.

The model assumes that input source signals have been recorded at a fixed
distance and with a fixed rotational speed. Auralisations are created
assuming free-field propagation and account for the following effects:
* Distance-dependent atmospheric absorption of high frequencies
* Doppler effect
* Ground reflection (including path length difference and
    angle-dependent high-frequency absorption)
* Distance-based amplitude panning (DBAP) to spatialise source in 3D
    loudspeaker array (direct sound and ground reflection spatialised
    separately).

Author: Marc C. Green (m.c.green@salford.ac.uk)
Affiliation: University of Salford (Acoustics Research Centre)
Copyright statement: This file and code is part of work undertaken within
the REFMAP project (www.refmap.eu), and is subject to licence as detailed
in the code repository.
https://github.com/acoustics-code-salford/uas-sound-event

Based on method detailed in:
Heutschi, K., Ott, B., Nussbaumer, T., and Wellig, P., 
"Synthesis of real world drone signals based on lab recordings,"
Acta Acustica, Vol. 4, No. 6, 2020, p. 24. 
https://doi.org/10.1051/aacus/2020023
'''