import pytest

from carscanner.models.peak import Peak
from carscanner.analysis.engineEstimator import EngineEstimator


def test_engineEstimator_estimate_findsRpmFromHarmonics():
    # Arrange: fire harmoniske af 22,5 Hz og én fremmed top ved 19,8 Hz.
    # amplitude og above_floor_db passer sammen med et tænkt gulv på 0,0001:
    # 20 * log10(0,001 / 0,0001) = 20 dB. Kammen bruger kun frekvensen.
    frequencies = [19.8, 67.5, 90.0, 157.5, 202.5]
    peaks = []
    for f in frequencies:
        peaks.append(Peak(frequency=f, amplitude=0.001, above_floor_db=20.0))

    # Act
    engine = EngineEstimator(cylinders=3).estimate(peaks, rpm_min=600, rpm_max=1200)

    # Assert: f_t = 22,5 Hz, rpm = 40 * 22,5 = 900, og 19,8 Hz er sorteret fra
    assert engine.firing_frequency == pytest.approx(22.5)
    assert engine.revolutions_per_minute == pytest.approx(900.0)
    assert [p.frequency for p in engine.matched_peaks] == [67.5, 90.0, 157.5, 202.5]