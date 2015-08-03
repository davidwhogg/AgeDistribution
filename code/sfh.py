import numpy as np
import matplotlib.pylab as plt
import emcee

class SFHModel():

    def __init__(self):
        """
        Note magic number 13.7 (age of the Universe in Gyr).
        Note many other magic numbers!
        """
        self.dlntime = 0.3
        self.lntimes = np.arange(np.log(0.1), np.log(13.7), self.dlntime)
        self.times = np.exp(self.lntimes)
        self.M = len(self.lntimes)
        self.prior_mean = np.log((20000. / 13.7) * self.times)
        self.prior_covar = 1. * np.exp(-1. * np.abs(np.arange(self.M)[:, None] -
                                                    np.arange(self.M)[None, :]))
        self.prior_invcovar = np.linalg.inv(self.prior_covar)
        assert np.all(np.linalg.eigvalsh(self.prior_invcovar) > 0.)

    def __call__(self, parvec):
        return self.lnprob(parvec)

    def set_data(self, ages, ivars):
        assert ivars.shape == ages.shape
        self.N = ages.shape[0]
        assert len(ages) == self.N
        assert np.all(ages > 0.)
        self.ages = ages
        self.ivars = ivars

    def load_data(self):
        """
        brittle function to read MKN data file

        HORRIBLE np.abs() CALL
        """
        fn = "../data/HWR_redclump_sample.txt"
        print "Reading %s ..." % fn
        data = np.genfromtxt(fn)
        names = ["ID", "distance", "Radius_gal", "Phi_gal", "z", "Teff",
                 "logg", "[Fe/H]", "[alpha/Fe]", "age"]
        print data[2]
        print data.shape
        print "Read %s" % fn
        ages = np.abs(data[:, 9]).flatten() # AARGH
        ivars = np.zeros_like(ages) + (0.25 / np.log10(np.e)) # MAGIC
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
        print "ln prior", lnp
        if not np.isfinite(lnp):
            return -np.Inf
        lnl = self.lnlike(parvec)
        print "ln like", lnl
        if not np.isfinite(lnl):
            return -np.Inf
        return lnp + lnl

    def lnprior(self, parvec):
        """
        proper prior from Gaussian processes
        """
        deltas = parvec - self.prior_mean
        return -0.5 * np.dot(deltas,
                             np.dot(self.prior_invcovar,
                                    deltas))

    def lnlike(self, parvec):
        """
        Poisson-like likelihood function.
        """
        return (np.sum(np.log(self.sfr(self.get_ages(), parvec,
                                      self.get_age_ivars())))
                - self.sfr_integral(lnamps))

if __name__ == "__main__":
    import os

    model = SFHModel()
    model.load_data()
    lnamps = 1. * model.prior_mean
    print model(lnamps)

    nwalkers = 4 * model.M
    p0 = model.prior_mean[None, :] +\
        0.001 * np.random.normal(size=(nwalkers, model.M))
    sampler = emcee.EnsembleSampler(nwalkers, model.M, model)

    newpid = 0
    finetgrid = np.exp(np.arange(np.log(0.05), np.log(13.8), 0.01))
    fineivars = np.zeros_like(finetgrid) + 0.25 / np.log10(np.e)
    for i in range(16):
        print "parent", i, newpid, "running emcee"
        sampler.run_mcmc(p0, 32)
        p0 = sampler.chain[:, -1, :].reshape(nwalkers, model.M)
        newpid = os.fork()
        if newpid == 0:
            plt.clf()
            for i in range(16):
                plt.plot(model.times, p0[i, :], "k-", alpha=0.5)
                plt.plot(finetgrid,
                         np.log(model.sfr(np.log(finetgrid), p0[i, :], fineivars)),
                         "k-", alpha=0.5)
            plt.plot(model.times, model.prior_mean, "r-", alpha=0.5)
            plt.plot(finetgrid,
                     np.log(model.sfr(np.log(finetgrid), model.prior_mean, fineivars)),
                         "r-", alpha=0.5)
            plt.semilogx()
            plt.xlabel("time (Gyr)")
            plt.ylabel("ln (number per ln time)")
            fn = "sfh.png"
            print "child saving", fn
            plt.savefig(fn)
            exit(0)
