import numpy as np
import pytest

from carscanner.models.spectrum import Spectrum
from carscanner.analysis.peakDetector import PeakDetector


def test_peakDetector_detect_findsOnlyPeaksAboveThreshold():
    # Arrange
    f0 = 0.2
    frequencies = np.arange(2501) * f0           # 0 til 500 Hz
    amplitudes = np.full(2501, 0.01)             # fladt gulv på 0,01
    amplitudes[450] = 0.1                        # 90 Hz,  20 dB over gulvet
    amplitudes[1000] = 0.02                      # 200 Hz,  6 dB over gulvet
    spectrum = Spectrum(frequencies=frequencies, amplitudes=amplitudes, f0=f0)
    floor = Spectrum(frequencies=frequencies, amplitudes=np.full(2501, 0.01), f0=f0)

    # Act
    peaks = PeakDetector(threshold_db=8, min_distance_hz=2).detect(spectrum, floor)

    # Assert: kun toppen ved 90 Hz er over tærsklen på 8 dB
    assert len(peaks) == 1
    assert peaks[0].frequency == pytest.approx(90.0)
    assert peaks[0].above_floor_db == pytest.approx(20.0)