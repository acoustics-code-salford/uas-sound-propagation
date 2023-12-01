'''
Summary
-------

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

**Author**: Marc C. Green (m.c.green@salford.ac.uk)

**Affiliation**: University of Salford (Acoustics Research Centre)

**Copyright statement**: This file and code is part of work undertaken within
the REFMAP project (www.refmap.eu), and is subject to license as detailed
in the [code repository
](https://github.com/acoustics-code-salford/uas-sound-event).

Based on method detailed in: [Heutschi, K., Ott, B., Nussbaumer, T., and
Wellig, P., "Synthesis of real world drone signals based on lab recordings,"
Acta Acustica, Vol. 4, No. 6, 2020, p. 24.
](https://doi.org/10.1051/aacus/2020023)

Usage
-----
The main object implementing the propagation model is `UASEventRenderer`. This
object is initialised using CSV files defining flightpaths in segments. Each
segment requires the definition of start and end points in cartesian
co-ordinates, along with starting and ending speeds in meters per second.
The following is an example flightpath configuration file with four segments.

```
segment, startxyz, endxyz, speeds
constant-speed flyby, 10 -200 30, 10 180 30, 30 30
decelerate, 10 180 30, 10 200 30, 30 0
accelerate, 10 200 30, 10 180 30, 0 30
flyby back, 10 180 30, 10 -200 30, 30 30
```

Using these CSV flightpath definitions, `UASEventRenderer` calculates the
position of the source at every sample time. These positions are used to
calculate the delays, distances, and angles required to render the propagation
effects based on a listener located at the origin of the co-ordinate system.

The listener height is set to 1.5 m by default. This affects the calculation
of the ground reflection signal in particular. Additional parameters include
the ground material, which can be selected from `'grass'`, `'soil'`, or
`'asphalt'`, and the loudspeaker layout to target with the DBAP panning
(currently only `'Octagon + Cube'` is implemented).

The object's `render()` method can then be used to generate a time-series
signal at the receiver point. The input source used should be sufficient in
length to cover the time taken to cover the entire defined flightpath.

The following is a basic script for rendering an event:

```
import soundfile as sf
from uasevent.environment import UASEventRenderer
from uasevent.utils import load_params

x, fs = sf.read('example_source.wav')
renderer = UASEventRenderer(
    load_params('flightpath.csv'),
    receiver_height=1.5,
    fs=fs)

output = renderer.render(x)
```

'''
