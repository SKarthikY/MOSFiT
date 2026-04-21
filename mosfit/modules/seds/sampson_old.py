"""sampson is the name of Karthik's emulator that produces SESN SEDs."""
from math import pi
import torch
import numpy as np
from astropy import constants as c
from astropy import units as u
from mosfit.constants import FOUR_PI
from mosfit.modules.seds.sed import SED

class Sampson(SED):
    """Stripped-Envelope spectral energy distribution. 
    This comes from Karthik's thesis work (see, e.g. https://arxiv.org/abs/2507.10648). 
    
    As of Nov 7 2025, sampson is still being trained, so I will write a trivial, fake version of this SED for now. 
    Once sampson is trained, I will come back and actually call the neural network. 
    sampson takes in 10 inputs: 
    inner velocity, outer velocity, velocity latent variable,
    nickel mass, nickel dist latent variable,
    helium mass, helium dist latent variable,
    opacity mass, opacity dist latent variable,
    time
    

    sampson spits out fluxes over a range of fixed wavelengths. 
    We can return both wavelenths and fluxes here, for good measure.
    """
    C_CONST = c.c.cgs.value
    FLUX_CONST = FOUR_PI * (
        2.0 * c.h * c.c ** 2 * pi).cgs.value * u.Angstrom.cgs.scale
    X_CONST = (c.h * c.c / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value

    N_wavelengths = 602
    wav_min = 2500
    wav_max = 12000
    fixed_wav_grid = np.linspace(wav_min+200, wav_max-300, N_wavelengths)
    
    mean_std_dict = None
    model = None
    _loaded = False

    def __init__(self, *args, **kwargs):
        """Initialize module."""
        super().__init__(*args, **kwargs)

        if not Sampson._loaded:
            print("Loading Sampson emulator...")
            Sampson.mean_std_dict = torch.load("normalization_stats_tmin4_tmax60_N15.pt", weights_only=False)
            Sampson.model = torch.load("emulator.pt", map_location = torch.device("cpu"), weights_only=False)


            Sampson.model.eval()
            Sampson._loaded = True

    print("hello1 from sampson SED!")

    def process(self, **kwargs):
        self._times = kwargs[self.key('dense_times')]
        self._dense_indices = kwargs[self.key('dense_indices')]


        lum_key = self.key('luminosities')
        kwargs = self.prepare_input(lum_key, **kwargs)
        self._luminosities = kwargs[lum_key]

        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._frequencies = kwargs['all_frequencies']


        self._eta_vel = kwargs[self.key('eta_vel')]
        self._min_vel = kwargs[self.key('min_vel')]
        self._del_vel = kwargs[self.key('del_vel')]

        self._eta_ni = kwargs[self.key('eta_ni')]
        self._eta_he = kwargs[self.key('eta_he')]
        self._eta_op = kwargs[self.key('eta_op')]

        self._mni = kwargs[self.key('m_ni')]
        self._mhe = kwargs[self.key('m_he')]
        self._mop = kwargs[self.key('m_op')]

        def_name = f'etavel{self._eta_vel}_minvel{self._min_vel}_delvel{self._del_vel}_etani{self._eta_ni}_etahe{self._eta_he}_etaop{self._eta_op}_mni{self._mni}_mhe{self._mhe}_mop{self._mop}'

        unnorm_descriptor = torch.tensor([
        self._eta_vel,
        self._eta_he,
        self._eta_ni,
        self._eta_op,
        self._min_vel,
        self._del_vel,
        self._mhe,
        self._mni,
        self._mop], dtype=torch.float32)
        xc = self.X_CONST  # noqa: F841
        fc = self.FLUX_CONST  # noqa: F841
        cc = self.C_CONST


        norm_descriptor = (unnorm_descriptor-Sampson.mean_std_dict['descriptor_mean'])/Sampson.mean_std_dict['descriptor_std']
        norm_descriptor_t = norm_descriptor.view(1, 9)  # (1, 9)

        #evaluate the neural network before the loop
        unique_t_indices = np.unique(self._dense_indices)
        unique_times = self._times[unique_t_indices]  # ndarray of times

        # 2) Normalize all those times and build batched input
        time_secs = unique_times * 86400.0
        norm_times = (
            time_secs - Sampson.mean_std_dict['time_mean']
        ) / Sampson.mean_std_dict['time_std']
        norm_times_t = torch.tensor(norm_times, dtype=torch.float32).view(-1, 1)  # (N_unique, 1)
        descriptor_batch = norm_descriptor_t.expand(norm_times_t.shape[0], -1)     # (N_unique, 9)
        input_batch = torch.cat((descriptor_batch, norm_times_t), dim=1).to(torch.float32)  # (N_unique, 10)

        # 3) Single batched NN call
        with torch.no_grad():
            pred_log_fluxes_batch = (
                Sampson.model(input_batch).to(torch.float64) * Sampson.mean_std_dict['fluxes_std']
                + Sampson.mean_std_dict['fluxes_mean']
            )  # (N_unique, N_wavelengths)


        pred_fluxes_batch = 10.0 ** pred_log_fluxes_batch
        pred_fluxes_batch_np = pred_fluxes_batch.detach().cpu().numpy()  # (N_unique, N_wavelengths)

        # 4) Build a map from dense index → predicted flux vector
        # unique_t_indices[k] corresponds to pred_fluxes_batch_np[k, :]
        flux_by_dense_index = {ti: pred_fluxes_batch_np[k]
                            for k, ti in enumerate(unique_t_indices)}


        # Some temp vars for speed.
        zp1 = 1.0 + kwargs[self.key('redshift')]
        Azp1 = u.Angstrom.cgs.scale / zp1
        czp1 = cc / zp1
        
        seds = []
        rest_wavs_dict = {}
        evaled = False
        dense_luminosities = []

        #print("in sampson, shape of sample_wavelengths: "+str(self._sample_wavelengths.shape))
        #print("in sampson, shape of self._times: "+str(self._times.shape))
        #print("in sampson, shape of self._luminosities: "+str(self._luminosities.shape))
        #print("in sampson, self._luminosities: "+str(self._luminosities))
        for li, lum in enumerate(self._luminosities):
            
            bi = self._band_indices[li]
            ti = self._dense_indices[li]
            dense_time = self._times[ti]


            if dense_time < 4 or dense_time > 60: # the earliest time I fit for. 
                seds.append(np.zeros(len(
                    self._sample_wavelengths[bi]) if bi >= 0 else 1))
                continue

            if bi >= 0:
                rest_wavs = rest_wavs_dict.setdefault(
                    bi, self._sample_wavelengths[bi] * Azp1)
            else:
                rest_wavs = np.array(  # noqa: F841
                    [czp1 / self._frequencies[li]])

            

            pred_fluxes_np = flux_by_dense_index[ti]

            rest_fluxes = np.interp(rest_wavs,
                        Sampson.fixed_wav_grid,
                        pred_fluxes_np)


            seds.append(rest_fluxes)
        seds = np.asarray(seds)
        seds = np.nan_to_num(seds, nan=0.0, posinf=0.0, neginf=0.0)
        seds[seds < 0] = 0.0              # no negative fluxes
        floor = 1e-40
        seds[seds == 0.0] = floor        # avoid exact zeros → -inf mags
        
        seds = self.add_to_existing_seds(seds, **kwargs)
        
        '''print("\ndef: "+str(def_name))
        print("infinities in seds: "+str(np.isinf(seds).sum()))
        print("NaNs in seds: " + str(np.isnan(seds).sum()))
        print("seds: "+str(seds))'''
        tor = {
            'sample_wavelengths':self._sample_wavelengths,
            self.key('seds'):seds,
            'times_out':self._times
        }


        return tor