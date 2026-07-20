"""
detector.py
Detector characteristics and saturation checks.
"""

from pathlib import Path
import numpy as np
from spectral_utils import load_two_column_curve


def load_qe_curve(path, wavelength_unit="Angstrom"):
    """
    Load a 2-column QE curve file (wavelength, QE fraction).

    Used for automatic loading of qe.dat from the data directory (no file
    dialog).  The curve unit is supplied explicitly by the configuration and
    converted to the common internal Angstrom representation.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    ndarray shape (n, 2)

    Raises
    ------
    FileNotFoundError if the file does not exist.
    ValueError if no valid rows are found.
    """
    curve = load_two_column_curve(path, wavelength_unit_name=wavelength_unit, name="QE curve")
    if np.any(curve.values < 0):
        raise ValueError("QE curve cannot contain negative values.")
    return curve.data


def load_gain_table(path):
    """Read a CMOS gain table: gain_setting, e-/ADU, read noise [e-], full well [e-].

    Amateur CMOS cameras change conversion gain, read noise and full well
    together with the gain setting; manufacturers publish these curves.  The
    file holds one whitespace- or comma-separated row per setting, ``#``
    comments allowed.  Returns rows sorted by gain setting.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Gain table not found: {path}")
    rows = []
    import re
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = re.split(r"[\s,;]+", line)
            if len(tokens) < 4:
                continue
            try:
                setting, e_adu, rn, fwc = (float(tokens[0]), float(tokens[1]),
                                           float(tokens[2]), float(tokens[3]))
            except ValueError:
                continue
            if e_adu <= 0 or rn < 0 or fwc <= 0:
                raise ValueError(f"Gain table row with non-physical values in {path}: {line!r}")
            rows.append({"gain_setting": setting, "gain_e_adu": e_adu,
                         "read_noise_e": rn, "full_well_e": fwc})
    if not rows:
        raise ValueError(f"No usable gain-table rows (setting, e-/ADU, RN, FWC) found in {path}")
    return sorted(rows, key=lambda row: row["gain_setting"])


def load_transmission_curve(path, wavelength_unit="Angstrom"):
    """Load a two-column wavelength/transmission curve in an explicit unit."""
    curve = load_two_column_curve(path, wavelength_unit_name=wavelength_unit, name="transmission curve")
    return curve.data


class Detector:
    """
    Detector parameters and utilities.
    
    Parameters
    ----------
    pixel_size_um : float
        Pixel size [μm]
    gain_e_adu : float
        Gain [electrons/ADU]
    full_well_e : float
        Full well capacity [electrons]
    bit_depth : int
        ADC bit depth (12, 14, 16, etc.)
    """
    
    # Fraction of the Bayer mosaic belonging to each channel (RGGB and
    # equivalents): light landing on the other channels' pixels is lost for
    # a single-channel extraction, and only that fraction of the pixels in
    # an aperture belongs to the channel.
    OSC_CHANNEL_FRACTION = {"R": 0.25, "G": 0.50, "B": 0.25}

    def __init__(self, pixel_size_um, gain_e_adu, full_well_e, bit_depth,
                 read_noise_e=5.0, dark_current_e_s_pix=0.0,
                 sensor_type="mono", osc_channel="G"):
        self.pixel_size_um = pixel_size_um
        self.gain_e_adu = gain_e_adu
        self.full_well_e = full_well_e
        self.bit_depth = bit_depth
        self.read_noise_e = float(read_noise_e)
        self.dark_current_e_s_pix = float(dark_current_e_s_pix)
        if self.read_noise_e < 0 or self.dark_current_e_s_pix < 0:
            raise ValueError("Read noise and dark current must be non-negative.")
        # Accept the user-facing labels 'monochrome'/'color' as aliases of the
        # internal 'mono'/'osc'.
        sensor = str(sensor_type).strip().lower()
        sensor = {"monochrome": "mono", "colour": "osc", "color": "osc"}.get(sensor, sensor)
        self.sensor_type = sensor
        if self.sensor_type not in {"mono", "osc"}:
            raise ValueError("Sensor type must be 'mono'/'monochrome' or 'osc'/'color'.")
        self.osc_channel = str(osc_channel).strip().upper()
        if self.sensor_type == "osc" and self.osc_channel not in self.OSC_CHANNEL_FRACTION:
            raise ValueError("OSC channel must be R, G, or B.")
        self.max_adu = 2**bit_depth - 1
        self.max_electrons = self.max_adu * gain_e_adu

    @property
    def channel_fill_fraction(self):
        """Bayer-mosaic area fraction of the selected channel (1 for mono).

        Applied to aperture/extraction-integrated source and sky rates and
        to the channel-pixel count; the *peak pixel* is not scaled, because
        a channel pixel at the PSF centre still receives the full local flux
        through its own filter dye (the QE curve should be the channel's
        effective QE = sensor QE x CFA dye transmission).
        """
        if self.sensor_type == "osc":
            return self.OSC_CHANNEL_FRACTION[self.osc_channel]
        return 1.0
    
    def check_saturation(self, counts_e_array):
        """
        Check if pixels are saturated.
        
        Parameters
        ----------
        counts_e_array : float or array
            Electron counts
        
        Returns
        -------
        is_saturated : bool or array
            True where saturated
        percent_saturated : float
            Percentage of saturated pixels
        """
        is_saturated = counts_e_array >= self.full_well_e
        
        if isinstance(is_saturated, np.ndarray):
            percent_saturated = 100 * np.sum(is_saturated) / len(counts_e_array)
        else:
            percent_saturated = 100.0 if is_saturated else 0.0
        
        return is_saturated, percent_saturated
    
    def counts_to_adu(self, counts_e_array):
        """
        Convert electron counts to ADU with clipping.
        
        Parameters
        ----------
        counts_e_array : float or array
            Electron counts
        
        Returns
        -------
        adu_array : int or array
            ADU values, clipped to [0, max_adu]
        is_saturated : bool or array
            True where ADU would exceed max_adu
        """
        adu = counts_e_array / self.gain_e_adu
        is_saturated = (adu >= self.max_adu) | (counts_e_array >= self.full_well_e)
        adu_clipped = np.clip(adu, 0, self.max_adu)
        
        if isinstance(adu_clipped, np.ndarray):
            adu_clipped = adu_clipped.astype(int)
        else:
            adu_clipped = int(adu_clipped)
        
        return adu_clipped, is_saturated

    def saturation_flag(self, counts_e_array):
        """Return explicit full-well/ADC saturation flags before clipping."""
        counts = np.asarray(counts_e_array, dtype=float)
        full_well = counts >= self.full_well_e
        adc = counts / self.gain_e_adu >= self.max_adu
        flags = np.where(full_well & adc, "BOTH",
                         np.where(full_well, "FULL_WELL", np.where(adc, "ADC", "NONE")))
        return str(flags.item()) if flags.ndim == 0 else flags
    
    def read_noise_electrons(self, readout_speed='slow'):
        """
        Estimate read noise.
        
        This is a placeholder; actual read noise depends on camera.
        
        Parameters
        ----------
        readout_speed : {'slow', 'medium', 'fast'}
        
        Returns
        -------
        read_noise_e : float
            Read noise [electrons RMS]
        """
        # Rough estimates
        read_noise_table = {
            'slow': 3.0,
            'medium': 5.0,
            'fast': 10.0
        }
        return read_noise_table.get(readout_speed, 5.0)
    
    def info(self):
        """Return detector information as dict."""
        return {
            'pixel_size_um': self.pixel_size_um,
            'gain_e_adu': self.gain_e_adu,
            'full_well_e': self.full_well_e,
            'bit_depth': self.bit_depth,
            'read_noise_e': self.read_noise_e,
            'dark_current_e_s_pix': self.dark_current_e_s_pix,
            'max_adu': self.max_adu,
            'max_electrons': self.max_electrons
        }
