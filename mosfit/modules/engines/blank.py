"""Definitions for the `NickelCobalt` class."""
import numpy as np

from mosfit.modules.engines.engine import Engine


# Important: Only define one ``Module`` class per file.


class Blank(Engine):
    """Blank engine. This was created for sampson"""

    def process(self, **kwargs):
        """Process module."""
        self._times = kwargs[self.key('dense_times')]
        if self.key('resttexplosion') in kwargs:
            self._rest_t_explosion = kwargs[self.key('resttexplosion')]
        else:
            # fall back to texplosion if resttexplosion isn't present
            self._rest_t_explosion = kwargs[self.key('texplosion')]

        # From 1994ApJS...92..527N
        ts = np.empty_like(self._times)
        t_inds = self._times >= self._rest_t_explosion
        ts[t_inds] = self._times[t_inds] - self._rest_t_explosion

        luminosities = np.zeros_like(self._times)

        luminosities[np.isnan(luminosities)] = 0.0

        return {self.dense_key('luminosities'): luminosities}
