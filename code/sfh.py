import numpy as np

class SFHModel():

    def __init__(self):
        """
        Note magic number 13.7 (age of the Universe in Gyr).
        Note many other magic numbers!
        """
        self.dlntime = 0.25
        self.lntimes = np.arange(np.log(0.01), np.log(13.7), self.dlntime)
        self.times = np.exp(self.lntimes)
        self.M = len(self.lntimes)
        self.prior_mean = np.log((20000. / 13.7) * self.times)
        self.prior_covar = 1. * np.exp(-1. * np.abs(np.arange(self.M)[:, None] -
                                         np.arange(self.M)[None, :]))
        self.prior_invcovar = np.linalg.inv(self.prior_covar)
        print np.sum(np.exp(self.prior_mean) * self.dlntime)

    def __call__(self, parvec):
        return self.lnprob(parvec)

    def set_data(self, ages, ivars):
        assert ivars.shape == ages.shape
        self.N = ages.shape[0]
        assert len(ages) == self.N
        self.ages = ages
        self.ivars = ivars

    def load_data(self):
        """
        brittle function to read MKN data file
        """
        fn = "../data/HWR_redclump_sample.txt"
        print "Reading %s ..." % fn
        data = np.genfromtxt(fn)
        names = ["ID", "distance", "Radius_gal", "Phi_gal", "z", "Teff",
                 "logg", "[Fe/H]", "[alpha/Fe]", "age"]
        print data[2]
        print data.shape
        print "Read %s" % fn
        ages = (data[:, 9]).flatten()
        ivars = np.zeros_like(ages) + (0.25 / np.log10(np.e))
        self.set_data(ages, ivars)

    def get_ages(self):
        return self.ages

    def get_age_ivars(self):
        return self.ivars

    def _oned_gaussian(self, x, mu, ivar):
        return np.exp(-0.5 * (x - mu) * (x - mu) * ivar) *\
            np.sqrt(0.5 * ivar / np.pi)

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
        deltas = parvec - self.prior_mean
        return np.dot(deltas,
                      np.dot(self.prior_invcovar,
                             deltas))

    def lnlike(self, parvec):
        """
        Poisson-like likelihood function.
        """
        lnamps = parvec
        return np.sum(np.log(self.sfr(self.get_ages(), lnamps,
                                      self.get_age_ivars()))) \
                                      - self.sfr_integral(lnamps)

if __name__ == "__main__":
    model = SFHModel()
    model.load_data()
    lnamps = 1. * model.prior_mean
    print model(lnamps)
