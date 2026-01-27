"""samson is the name of Karthik's emulator that produces SESN SEDs."""
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
    

    samson spits out fluxes over a range of fixed wavelengths. 
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

    mean_std_dict = torch.load("/n/home07/kyadavalli/scratch/NeuralNetworks/HOperation/smoothing_spectra/normalization_stats_normwindows3_windowlength30_polyorder4.pt", weights_only=False)
    
    model = torch.load("/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/training/model_ckpt/dim256_nhead16_numlayers7_learnedPETrue_lr0.0001_weightdecay0.01_batchsize32_epochs3000_window_length_90/100.pth", map_location = torch.device("cpu"), weights_only=False)
    model.eval()

    print("hello1 from sampson SED!")

    def process(self, **kwargs):
        self._times = kwargs[self.key('rest_times')]
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

        
        norm_descriptor = (unnorm_descriptor-self.mean_std_dict['descriptor_mean'])/self.mean_std_dict['descriptor_std']
        #print("got dense_times: "+str(self._times))
        #print("got self_wavelengths: "+str(self._sample_wavelengths))
        #print("band_indices: "+str(self._band_indices))

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
            if self._times.shape != self._luminosities.shape or len(self._luminosities) == 870:
                seds.append(np.zeros(len(
                    self._sample_wavelengths[bi]) if bi >= 0 else 1))
                continue

            dense_time = self._times[li]
            #print("dense_time: "+str(dense_time))
            if dense_time < 7 or dense_time > 60: # the earliest time I fit for. 
                seds.append(np.zeros(len(
                    self._sample_wavelengths[bi]) if bi >= 0 else 1))
                continue

            if bi >= 0:
                rest_wavs = rest_wavs_dict.setdefault(
                    bi, self._sample_wavelengths[bi] * Azp1)
            else:
                rest_wavs = np.array(  # noqa: F841
                    [czp1 / self._frequencies[li]])

            

            time_sec = float(dense_time*86400)
            norm_time = (time_sec - self.mean_std_dict['time_mean'])/self.mean_std_dict['time_std']
            norm_time_t = torch.tensor(norm_time, dtype=torch.float32).view(1,1)

            # make descriptor (1,9)
            norm_descriptor_t = norm_descriptor.view(1, 9)

            input_data = torch.cat((norm_descriptor_t, norm_time_t), dim=1)
            input_data = input_data.to(torch.float32)

            pred_fluxes = 10.**((self.model(input_data).to(torch.float64)*self.mean_std_dict['fluxes_std'])+self.mean_std_dict['fluxes_mean'])
            pred_fluxes_np = pred_fluxes.detach().cpu().numpy().flatten()

            rest_fluxes = np.interp(rest_wavs,
                        Samson.fixed_wav_grid,
                        pred_fluxes_np)
            #print("rest fluxes: "+str(rest_fluxes))
            seds.append(rest_fluxes)
        seds = np.asarray(seds)
        seds = self.add_to_existing_seds(seds, **kwargs)
        print("\ndef: "+str(def_name))
        print("infinities in seds: "+str(np.isinf(seds).sum()))
        print("NaNs in seds: " + str(np.isnan(seds).sum()))
        print("seds: "+str(seds))
        tor = {
            'sample_wavelengths':self._sample_wavelengths,
            self.key('seds'):seds,
            'times_out':self._times
        }


        return tor




            

            


                    
                    




