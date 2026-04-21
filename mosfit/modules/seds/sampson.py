"""
sampson.py

MOSFiT SED module wrapping Karthik's stripped-envelope SN emulator ("sampson").

This class:
- builds a descriptor vector from MOSFiT parameters,
- normalizes descriptor and time using precomputed stats,
- calls a trained neural emulator (SimpleFluxMLP),
- returns model SEDs on MOSFiT's wavelength grid.
"""

from math import pi
import numpy as np
import torch

from astropy import constants as c
from astropy import units as u

from mosfit.constants import FOUR_PI, KM_CGS, M_SUN_CGS
from mosfit.modules.seds.sed import SED

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFluxMLP(nn.Module):
    """
    Simple MLP mapping physical parameters -> flux spectrum.

    Args:
        n_physical_param: input dimension (number of physical parameters)
        n_wavelength: output dimension (number of wavelength points)
        d_model: hidden dimension
        num_layers: total number of linear layers (>= 2)
        nhead, learnedPE: kept for API compatibility, not used
    """
    def __init__(self,
                 n_physical_param=10,
                 n_wavelength=602,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 learnedPE=True):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2 for an input and output layer."

        layers = []

        # Input layer
        layers.append(nn.Linear(n_physical_param, d_model))
        layers.append(nn.GELU())

        # Hidden layers: num_layers - 2 internal linear layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.GELU())

        # Output layer
        layers.append(nn.Linear(d_model, n_wavelength))

        self.net = nn.Sequential(*layers)

    def forward(self, physical_param):
        """
        physical_param: [B, n_physical_param]
        Returns: [B, n_wavelength]
        """
        return self.net(physical_param)


# ---------------------------------------------------------------------------
# User-adjustable paths
# ---------------------------------------------------------------------------

NORMALIZATION_STATS_PATH = (
    "/n/home07/kyadavalli/kyadavalli/installed_packages/MOSFiT/"
    "mosfit/modules/seds/normalization_stats_tmin4_tmax60_N15.pt"
)

EMULATOR_WEIGHTS_PATH = (
    "/n/home07/kyadavalli/kyadavalli/installed_packages/MOSFiT/"
    "mosfit/modules/seds/emulator.pth"
)

# ---------------------------------------------------------------------------
# SED implementation
# ---------------------------------------------------------------------------


class Sampson(SED):
    """Stripped-envelope spectral energy distribution from NN emulator."""
    
    # Physical constants (cgs)
    C_CONST = c.c.cgs.value
    FLUX_CONST = (
        FOUR_PI * (2.0 * c.h * c.c ** 2 * pi).cgs.value * u.Angstrom.cgs.scale
    )
    X_CONST = (c.h * c.c / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    EMULATOR_T_MIN = 4.0
    EMULATOR_T_MAX = 55.0   # ← was 60.0, buffer against edge effects

    # Wavelength grid used by the emulator.
    # Your preprocessing used 602 points from 2200–9700 Å, then N=15 downsampling.
    # 602 -> indices 0,15,...,600 → 41 points.
    _orig_wav_grid = np.linspace(2000.0 + 200.0, 10000.0 - 300.0, 602)
    fixed_wav_grid = _orig_wav_grid[::15]  # length 41
    _debug_call_count = 0

    # Shared state
    mean_std_dict = None
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _loaded = False

    def __init__(self, *args, **kwargs):
        """Initialize module and lazily load emulator + stats once."""
        super().__init__(*args, **kwargs)

        if not Sampson._loaded:
            print("Loading Sampson emulator on device:", Sampson.device)

            # ------------------------------------------------------------------
            # Load normalization statistics (mean/std for descriptor, time, flux)
            # ------------------------------------------------------------------
            msd = torch.load(
                NORMALIZATION_STATS_PATH,
                map_location=Sampson.device,
                weights_only=False,  # needed in PyTorch 2.6+ for non-state_dict
            )
            if not isinstance(msd, dict):
                raise RuntimeError(
                    f"Expected normalization stats at {NORMALIZATION_STATS_PATH} "
                    "to be a dict of tensors (means/stds). Got type: "
                    f"{type(msd)}"
                )

            # Move tensor values to device as float32
            for k, v in list(msd.items()):
                if torch.is_tensor(v):
                    msd[k] = v.to(device=Sampson.device, dtype=torch.float32)

            # Basic sanity checks
            assert "descriptor_mean" in msd and "descriptor_std" in msd, \
                "Stats dict must contain 'descriptor_mean' and 'descriptor_std'."
            assert "time_mean" in msd and "time_std" in msd, \
                "Stats dict must contain 'time_mean' and 'time_std'."
            assert "fluxes_mean" in msd and "fluxes_std" in msd, \
                "Stats dict must contain 'fluxes_mean' and 'fluxes_std'."

            # Descriptor stats
            desc_mean = msd["descriptor_mean"]
            desc_std = msd["descriptor_std"]
            assert desc_mean.shape == desc_std.shape == (9,), \
                f"Expected descriptor_mean/std shape (9,), got {desc_mean.shape}, {desc_std.shape}"

            # Flux stats are GLOBAL SCALARS in your script; no shape check here.
            # Broadcasting will apply mean/std to each wavelength bin.

            # Sanity: make sure our fixed_wav_grid and model n_wav agree
            n_wav = Sampson.fixed_wav_grid.shape[0]
            assert n_wav == 41, f"Expected 41 wavelength bins, got {n_wav}"

            Sampson.mean_std_dict = msd

            # ------------------------------------------------------------------
            # Build emulator and load weights (SimpleFluxMLP, not Transformer)
            # ------------------------------------------------------------------
            Sampson.model = SimpleFluxMLP(
                d_model=512,
                num_layers=10,
                n_wavelength=n_wav,
            ).to(Sampson.device)

            state_dict = torch.load(
                EMULATOR_WEIGHTS_PATH,
                map_location="cpu",  # state_dict is device-agnostic
                weights_only=True,   # this file is a pure state_dict
            )
            Sampson.model.load_state_dict(state_dict)
            Sampson.model.to(Sampson.device)
            Sampson.model.eval()
            torch.set_grad_enabled(False)

            def _safe_scalar(v):
                """Convert tensor or float to a printable Python scalar."""
                if torch.is_tensor(v):
                    return float(v.cpu())
                return float(v)

            print("=== NORMALIZATION STATS (printed once) ===")
            print("descriptor_mean:", msd["descriptor_mean"].cpu().numpy())
            print("descriptor_std: ", msd["descriptor_std"].cpu().numpy())
            print("time_mean:      ", _safe_scalar(msd["time_mean"]))
            print("time_std:       ", _safe_scalar(msd["time_std"]))
            print("fluxes_mean:    ", _safe_scalar(msd["fluxes_mean"]))
            print("fluxes_std:     ", _safe_scalar(msd["fluxes_std"]))


            Sampson._loaded = True

    def process(self, **kwargs):
        """Run the emulator and return SEDs on MOSFiT's wavelength grid."""

        # ----------------------------------------------------------------------
        # Extract common inputs from MOSFiT
        # ----------------------------------------------------------------------
        # Observer-frame dense times in days (typically MJD or days relative to ref)
        obs_times = kwargs[self.key("dense_times")]              # [N_dense]
        self._dense_indices = kwargs[self.key("dense_indices")]  # [N_points]

        # Explosion time parameter: must be in same units as obs_times (days)
        texp = kwargs[self.key("texplosion")]                    # scalar

        # Redshift
        z = kwargs[self.key("redshift")]

        # Rest-frame time since explosion in days (what the emulator was trained on)
        t_rest = (obs_times - texp) / (1.0 + z)

        if torch.is_tensor(texp):
            texp_val = float(texp)
        else:
            texp_val = texp
        #print("DEBUG Sampson.py: min(obs_times), max(obs_times) =", obs_times.min(), obs_times.max())
        #print("DEBUG Sampson.py: texp =", texp_val, "z =", float(z))
        #print("DEBUG Sampson.py: min(t_rest), max(t_rest) =", t_rest.min(), t_rest.max())

        self._times = t_rest    # use this as the "time axis" for the emulator

        lum_key = self.key("luminosities")
        kwargs = self.prepare_input(lum_key, **kwargs)
        self._luminosities = kwargs[lum_key]                     # [N_points]

        valid_mask = np.zeros(len(self._luminosities), dtype=bool)

        self._bands = kwargs["all_bands"]
        self._band_indices = kwargs["all_band_indices"]          # [N_points]
        self._frequencies = kwargs["all_frequencies"]            # [N_points]

        # Descriptor parameters from MOSFiT (must match training convention)
        self._eta_vel = kwargs[self.key("eta_vel")]
        #this will be passed in, in units of km/s. I need it in cm/s
        self._min_vel = kwargs[self.key("min_vel")] * 100 * KM_CGS
        self._del_vel = kwargs[self.key("del_vel")] * 100 * KM_CGS

        self._eta_ni = kwargs[self.key("eta_ni")]
        self._eta_he = kwargs[self.key("eta_he")]
        self._eta_op = kwargs[self.key("eta_op")]

        self._mni = kwargs[self.key("m_ni")] * M_SUN_CGS
        self._mhe = kwargs[self.key("m_he")] * M_SUN_CGS
        self._mop = kwargs[self.key("m_op")] * M_SUN_CGS

        # Unnormalized descriptor vector (9 parameters; same order as training)
        unnorm_descriptor = torch.tensor(
            [
                self._eta_vel,
                self._eta_he,
                self._eta_ni,
                self._eta_op,
                self._min_vel,
                self._del_vel,
                self._mhe,
                self._mni,
                self._mop,
            ],
            dtype=torch.float32,
            device=Sampson.device,
        )

        cc = self.C_CONST
        msd = Sampson.mean_std_dict

        # ----------------------------------------------------------------------
        # Normalize descriptor and prepare batched input
        # ----------------------------------------------------------------------
        desc_mean = msd["descriptor_mean"]  # shape [9]
        desc_std = msd["descriptor_std"]    # shape [9]

        norm_descriptor = (unnorm_descriptor - desc_mean) / desc_std
        norm_descriptor_t = norm_descriptor.view(1, 9)  # (1, 9)

        # Unique dense time indices (so we call NN once per unique time)
        unique_t_indices = np.unique(self._dense_indices)
        unique_times = self._times[unique_t_indices]  # days, ndarray

        # Normalize times (training used time in seconds)
        time_secs = unique_times * 86400.0
        time_secs_t = torch.tensor(
            time_secs,
            dtype=torch.float32,
            device=Sampson.device,
        )

        time_mean = msd["time_mean"]
        time_std = msd["time_std"]
        # Handle possible scalar vs 0-dim tensor
        if not torch.is_tensor(time_mean):
            time_mean = torch.tensor(time_mean, dtype=torch.float32, device=Sampson.device)
        if not torch.is_tensor(time_std):
            time_std = torch.tensor(time_std, dtype=torch.float32, device=Sampson.device)
        time_mean = time_mean.to(Sampson.device, dtype=torch.float32)
        time_std = time_std.to(Sampson.device, dtype=torch.float32)

        norm_times_t = (time_secs_t - time_mean) / time_std
        norm_times_t = norm_times_t.view(-1, 1)  # (N_unique, 1)

        # Repeat descriptor for each time and concatenate
        descriptor_batch = norm_descriptor_t.expand(norm_times_t.shape[0], -1)  # (N_unique, 9)
        input_batch = torch.cat((descriptor_batch, norm_times_t), dim=1)        # (N_unique, 10)

        # ----------------------------------------------------------------------
        # Single batched network evaluation
        # ----------------------------------------------------------------------
        
        valid_query_mask = (
            (unique_times >= Sampson.EMULATOR_T_MIN) & 
            (unique_times <= Sampson.EMULATOR_T_MAX)
        )

        Sampson._debug_call_count += 1
        if Sampson._debug_call_count <= 3:   # only print first 3 calls
            print(f"\n=== EMULATOR INPUT (call #{Sampson._debug_call_count}) ===")
            print("unnorm_descriptor:", unnorm_descriptor.cpu().numpy())
            print("  eta_vel  =", float(self._eta_vel))
            print("  eta_he   =", float(self._eta_he))
            print("  eta_ni   =", float(self._eta_ni))
            print("  eta_op   =", float(self._eta_op))
            print("  min_vel  =", float(self._min_vel), " (training units?)")
            print("  del_vel  =", float(self._del_vel), " (training units?)")
            print("  m_he     =", float(self._mhe),     " (training units?)")
            print("  m_ni     =", float(self._mni),     " (training units?)")
            print("  m_op     =", float(self._mop),     " (training units?)")
            print("norm_descriptor:  ", norm_descriptor.cpu().numpy())
            print("t_rest (days):    ", unique_times[:5], "...")
            print("norm_times:       ", norm_times_t.squeeze().cpu().numpy()[:5], "...")

            # Also print first predicted flux to check magnitude
            if np.any(valid_query_mask):
                with torch.no_grad():
                    test_out = (Sampson.model(input_batch[valid_query_mask][:1]) 
                            * msd["fluxes_std"] + msd["fluxes_mean"])
                    test_flux = 10.0 ** test_out
                print("pred log10(flux) at t=first valid time:", 
                    test_out[0, 20].item(), "(middle wavelength)")
                print("pred flux at t=first valid time:       ", 
                    test_flux[0, 20].item(), "(middle wavelength)")
                print("wavelength at index 20:                ", 
                    Sampson.fixed_wav_grid[20], "Angstrom")

        if np.any(valid_query_mask):
            valid_input = input_batch[valid_query_mask]
            with torch.no_grad():
                pred_valid = (
                    Sampson.model(valid_input) * msd["fluxes_std"] + msd["fluxes_mean"]
                )
            pred_fluxes_valid = 10.0 ** pred_valid
            pred_fluxes_valid_np = pred_fluxes_valid.detach().cpu().numpy()

        # Map back
        flux_by_dense_index = {}
        valid_idx = 0
        for k, ti in enumerate(unique_t_indices):
            if valid_query_mask[k]:
                flux_by_dense_index[ti] = pred_fluxes_valid_np[valid_idx]
                valid_idx += 1
            else:
                flux_by_dense_index[ti] = np.full(len(Sampson.fixed_wav_grid), 1e-40)

        # ----------------------------------------------------------------------
        # Build SEDs at each requested (time, band)
        # ----------------------------------------------------------------------
        zp1 = 1.0 + kwargs[self.key("redshift")]
        czp1 = cc / zp1

        seds = []
        rest_wavs_dict = {}

        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            ti = self._dense_indices[li]

            dense_time = self._times[ti]  # rest-frame days since explosion
            


            # mark whether this point is inside emulator's training window
            if Sampson.EMULATOR_T_MIN <= dense_time <= Sampson.EMULATOR_T_MAX:
                valid_mask[li] = True

            # After building valid_mask in sampson.py process():
            n_valid = valid_mask.sum()
            n_total = len(valid_mask)
            '''print(f"DEBUG: {n_valid}/{n_total} points inside valid window [4,60] days")
            if n_valid == 0:
                print("WARNING: NO VALID POINTS — emulator being called at wrong times!")'''

            # Rest-frame wavelengths for this band
            if bi >= 0:
                rest_wavs = rest_wavs_dict.setdefault(
                    bi,
                    self._sample_wavelengths[bi] / zp1,
                )
            else:
                rest_wavs = np.array([czp1 / self._frequencies[li]])

            # Emulator flux vector (on fixed_wav_grid) at this time index
            pred_fluxes_np = flux_by_dense_index[ti]

            # Interpolate emulator fluxes onto the band rest-frame wavelength grid
            rest_fluxes = np.interp(
                rest_wavs,
                Sampson.fixed_wav_grid,
                pred_fluxes_np,
            )

            # NOTE: `lum` is not currently used to rescale flux.

            seds.append(rest_fluxes)

        # ----------------------------------------------------------------------
        # Clean up fluxes and add to existing MOSFiT SEDs
        # ----------------------------------------------------------------------


        seds = np.asarray(seds)
        seds = np.nan_to_num(seds, nan=0.0, posinf=0.0, neginf=0.0)
        seds[seds < 0] = 0.0
        floor = 1e-40
        seds[seds == 0.0] = floor

        seds = self.add_to_existing_seds(seds, **kwargs)

        tor = {
            "sample_wavelengths": self._sample_wavelengths,
            self.key("seds"): seds,
            # per-point validity mask: True where 4 <= t_rest <= 60
            "sesn_valid_mask": valid_mask,
            # original observer-frame times for downstream modules
            "times_out": obs_times,
        }

        return tor