import numpy as np
import pytest

from carscanner.sources.signalGenerator import SignalGenerator
from carscanner.analysis.spectrumAnalyzer import SpectrumAnalyzer
from carscanner.analysis.window import HanningWindow


def test_spectrumAnalyzer_analyze_sineOnLineGivesCorrectPeak():
    # Arrange
    sine = SignalGenerator().sine(frequency=20, amplitude=1, duration=5, fs=48000)
    analyzer = SpectrumAnalyzer(HanningWindow())
    
    # Act
    spectrum = analyzer.analyse(signal=sine)
    m_peak = np.argmax(spectrum.amplitudes) # returner et array af de største amplitude værdier i spectrummet
    # Assert
    assert spectrum.frequencies[m_peak] == pytest.approx(20) # forventer at se amplituden 1 ved 20 Hz(dens top svingning)
    assert spectrum.amplitudes[m_peak] == pytest.approx(1.0, rel=1e-3) # forventer at den største værdi er 1

def test_spectrumAnalyzer_analyse_frequencyAxisIsCorrect():
    # Arrange
    sine = SignalGenerator().sine(frequency=20, amplitude=1, duration=5, fs=48000)
    analyzer = SpectrumAnalyzer(HanningWindow())

    # Act
    spectrum = analyzer.analyse(signal=sine)

    # Assert
    # f0 = 1/T = 1/5 s = 0,2 Hz mellem stregerne
    assert spectrum.f0 == pytest.approx(0.2)
    # N = T * f_s = 240 000 samples. rfft giver kun den positive halvdel:
    # N/2 + 1 streger, fordi både 0 Hz og f_s/2 er med.
    assert len(spectrum.frequencies) == 120001
    # Sidste streg er f_s/2, toppen af basebåndet
    assert spectrum.frequencies[-1] == pytest.approx(24000)
