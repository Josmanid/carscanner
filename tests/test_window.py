import numpy as np
import pytest

from carscanner.sources.signalGenerator import SignalGenerator
from carscanner.analysis.window import HanningWindow, RectangularWindow

# Tjekker, at Hanning går til 0 i begge ender af signalet i tid.
# Så er der intet spring, når FFT'en gentager signalet, og lækagen bliver mindre.
def test_hanningWindow_applyTo_endsAreZero():
    # Arrange
    # 18,75 Hz i 5 s er 93,75 svingninger, ikke et helt antal, så uden vindue
    # ville der være et spring. Amplitude 1 for enkelhedens skyld, 5 s som et
    # Welch-stykke og 48 000 Hz som optagelserne.
    signal = SignalGenerator().sine(frequency=18.75, amplitude=1, duration=5, fs=48000)

    # Act
    windowed_sine = HanningWindow().apply_to(signal=signal)

    # Assert
    # Første og sidste sample er 0. abs=1e-12 er en fast tolerance,
    # fordi facit er 0, og tallet kan være -0.0 eller en lille afrunding.
    assert windowed_sine.samples[0] == pytest.approx(0.0, abs=1e-12)
    assert windowed_sine.samples[-1] == pytest.approx(0.0, abs=1e-12)   


# Tjekker, at det rektangulære vindue ikke ændrer signalet.
# Det ganger med 1 overalt og bruges kun til at sammenligne med Hanning.
def test_rectangularWindow_applyTo_leavesSignalUnchanged():
    # Arrange
    signal = SignalGenerator().sine(frequency=18.75, amplitude=1, duration=5, fs=48000)

    # Act
    windowed_sine = RectangularWindow().apply_to(signal=signal)

    # Assert
    # np.array_equal giver True, hvis alle værdier i de to arrays er ens.
    assert np.array_equal(windowed_sine.samples, signal.samples)
