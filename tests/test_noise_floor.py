import numpy as np
import pytest

from carscanner.models.spectrum import Spectrum
from carscanner.analysis.noiseFloorEstimator import NoiseFloorEstimator

def test_noiseFloorEstimator_estimate_ignoresNarrowPeaks():

    # Arrange
    f0 = 0.2                                    # 0,2 Hz mellem stregerne
    frequencies = np.arange(2501) * f0          # 2501 streger fra 0 til 500 Hz
    amplitudes = np.full(2501, 0.01)            # numpy-array med 0,01 overalt = fladt gulv
    for f in (100, 200, 300):                   # tuplen (100, 200, 300): tre toppe
        m = int(f / f0)                         # stregens nummer, m = f / f0
        amplitudes[m - 2:m + 3] = 1.0           # 5 streger = én top på 1 Hz
    spectrum = Spectrum(frequencies=frequencies, amplitudes=amplitudes, f0=f0)
    # Act
    floor = NoiseFloorEstimator(window_hz=10).estimate(spectrum=spectrum)

    # Assert
    assert np.allclose(floor.amplitudes, 0.01)

def test_noiseFloorEstimator_estimate_followsRisingFloor():
    # Arrange
    f0 = 0.2                                        # 0,2 Hz mellem stregerne
    frequencies = np.arange(2501) * f0              # 2501 streger fra 0 til 500 Hz
    # Et gulv, der stiger jævnt fra 0,01 til 0,1, uden toppe.
    # linspace laver 2501 tal med lige stor afstand mellem start og slut.
    amplitudes = np.linspace(0.01, 0.1, 2501)
    spectrum = Spectrum(frequencies=frequencies, amplitudes=amplitudes, f0=f0)

    # Act
    floor = NoiseFloorEstimator(window_hz=10).estimate(spectrum=spectrum)

    # Assert
    # Medianen af en jævnt stigende række er den midterste værdi,
    # så gulvet skal følge kurven. De første og sidste 100 streger,
    # 20 Hz i hver ende, springes over, fordi mode="nearest" giver
    # en lille kanteffekt dér.
    assert np.allclose(floor.amplitudes[100:-100], amplitudes[100:-100])