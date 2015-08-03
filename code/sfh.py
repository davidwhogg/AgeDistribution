import numpy as np

class SFHModel():

    def __init__(self):
        """
        Note magic number 13.7 (age of the Universe in Gyr).
        """
        self.dlntime = 0.5
        self.lntimes = np.arange(np.log(0.01), np.log(13.7), self.dlntime)
        self.times = np.exp(self.lntimes)
        self.M = len(self.lntimes)
        print self.dlntime, self.lntimes, self.times, self.M

    def __call__(self, parvec):
        return self.lnprob(parvec)

    def set_data(self, ages, ivars):
        assert ivars.shape == ages.shape
        self.N = ages.shape[0]
        assert len(ages) == self.N
        self.ages = ages
        self.ivars = ivars

    def sfr(self, lntimes, lnamps, ivars):
        """
        The star-formation rate mean model is a mixture of delta
        functions, smoothed by a Gaussian (in ln time) of width ivar
        (in nats).

        This is a rate *per ln time* (per nat) not *per time* (per Gyr).

        input lntimes are np.log(times) and are times *ago* in Gyr.

        input ivars should match lntimes -- in principle each star has
        the SFH smoothed with its own, special ivar!
        """
        assert len(lnamps) == self.M
        assert len(lntimes) == len(ivars)
        rates = 0.
        for lnt, lnamp in zip(self.lntimes, lnamps):
            rates = rates + np.exp(lnamp) *\
                self._oned_gaussian(lntimes, lnt, ivars)
        return rates

    def sfr_integral(self, lnamps):
        """
        approximation: The model has support at times > the age of the
        Universe.
        """
        return np.sum(np.exp(lnamps))

    def lnprob(self, parvec):
        lnp = self.lnprior(parvec)
        if not np.isfinite(lnp):
            return -np.Inf
        return lnp + self.lnlike(parvec)

    def lnprior(self, parvec):
        """
        proper prior from Gaussian processes
        """
        lnamps = parvec
        mu = np.log(2000. / self.dlntime) + self.lntimes # fucked 2000 stars per Gyr?
        print mu
        covar = 1. * np.exp(-1. * np.abs(np.arange(self.M)[:, None] -
                                         np.arange(self.M)[None, :]))
        print covar
        assert False
        return np.dot((lnamps - mu), np.linalg.solve(covar, (lnamps - mu)))

    def lnlike(self, parvec):
        """
        Poisson-like likelihood function.
        """
        lnamps = parvec
        return np.sum(np.log(self.sfr(self.get_ages(), lnamps,
                                      self.get_age_ivars()))) \
                                      - sfr_integral(lnamps)

if __name__ == "__main__":
    model = SFHModel
