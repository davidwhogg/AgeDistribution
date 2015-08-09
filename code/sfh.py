"""
This file is part of the *AgeDistribution* project.
Copyright 2015 David W. Hogg (NYU).

## bugs:
- doesn't have an associated LaTeX document
- SFHModel needs __print__(), and plot() functions
- optimize first, sample later
"""

import numpy as np
from scipy.stats import norm as gaussian
import matplotlib.pylab as plt
import emcee

class SFHModel():

    def __init__(self):
        """
        Note magic number 13.7 (age of the Universe in Gyr).
        Note many other magic numbers!
        """
        self.dlntime = 0.3
        self.minage = 0.05 # Gyr
        self.maxage = 13.7 # Gyr
        self.minlnage = np.log(self.minage)
        self.maxlnage = np.log(self.maxage)
        self.lntimes = np.arange(self.minlnage, self.maxlnage, self.dlntime)
        self.times = np.exp(self.lntimes)
        self.M = len(self.lntimes)
        self.prior_mean = np.log((5000. / 13.7) * self.times)
        self.prior_covar = 1. * np.exp(-1. * np.abs(np.arange(self.M)[:, None] -
                                                    np.arange(self.M)[None, :]))
        self.prior_invcovar = np.linalg.inv(self.prior_covar)
        assert np.all(np.linalg.eigvalsh(self.prior_invcovar) > 0.)

    def __call__(self, parvec):
        return self.lnprob(parvec)

    def set_data(self, ages, ivars):
        assert ivars.shape == ages.shape
        good = (ages > self.minage) * (ages < self.maxage)
        self.ages = ages[good]
        self.ivars = ivars[good]
        self.N = self.ages.shape[0]
        assert len(self.ages) == self.N
        assert np.all(self.ages >= self.minage)
        assert np.all(self.ages <= self.maxage)
        assert np.all(self.ivars >= 0.)
        self.lnages = np.log(self.ages)

    def load_data(self):
        """
        brittle function to read MKN data file

        HORRIBLE clip.
        """
        fn = "../data/HWR_redclump_sample.txt"
        print "Reading %s ..." % fn
        data = np.genfromtxt(fn)
        names = ["ID", "distance", "Radius_gal", "Phi_gal", "z", "Teff",
                 "logg", "[Fe/H]", "[alpha/Fe]", "age"]
        print data[2]
        print data.shape
        print "Read %s" % fn
        ages = data[:, 9].flatten()
        sigma = 0.25 / np.log10(np.e) # MAGIC
        ivars = np.zeros_like(ages) + 1. / (sigma * sigma)
        self.set_data(ages, ivars)

    def get_lnages(self):
        return self.lnages

    def get_lnage_ivars(self):
        return self.ivars

    def _oned_gaussian(self, x, mu, ivar):
        return gaussian.pdf(x - mu) * np.sqrt(ivar)

    def _integrate_oned_gaussian(self, x1, x2, mu, ivar):
        return (gaussian.cdf(x2 - mu) -
                gaussian.cdf(x1 - mu)) * np.sqrt(ivar)

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
        rates = np.zeros_like(lntimes)
        for lnt, lnamp in zip(self.lntimes, lnamps):
            rates += np.exp(lnamp) * self._oned_gaussian(lntimes, lnt, ivars)
        return rates

    def sfr_integral(self, lnamps, ivars):
        """
        approximation: The model has support at times > the age of the
        Universe.
        """
        assert len(lnamps) == self.M
        meanivar = np.mean(ivars)
        integral = 0.
        for lnt, lnamp in zip(self.lntimes, lnamps):
            integral += np.exp(lnamp) * \
                self._integrate_oned_gaussian(self.minlnage, self.maxlnage,
                                              lnt, meanivar)
        return integral

    def lnprob(self, parvec):
        lnp = self.lnprior(parvec)
        if not np.isfinite(lnp):
            return -np.Inf
        lnl = self.lnlike(parvec)
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
        return (np.sum(np.log(self.sfr(self.get_lnages(), parvec,
                                       self.get_lnage_ivars())))
                - self.sfr_integral(parvec, self.get_lnage_ivars()))

if __name__ == "__main__":
    import os

    model = SFHModel()
    model.load_data()
    lnamps = 1. * model.prior_mean
    print model(lnamps)

    dNdlnt, bins = np.histogram(model.get_lnages(), bins=128,
                                range=(np.log(0.05), np.log(13.8)), density=True)
    dNdlnt *= float(model.N)
    xhist = np.exp((bins[:, None] * np.ones(2)[None, :]).flatten())[1:-1]
    yhist = (dNdlnt[:, None] * np.ones(2)[None, :]).flatten()
    yhist = np.clip(yhist, 1., np.Inf) # MAGIC

    nwalkers = 4 * model.M
    p0 = model.prior_mean[None, :] +\
        0.001 * np.random.normal(size=(nwalkers, model.M))
    sampler = emcee.EnsembleSampler(nwalkers, model.M, model)

    newpid = 0
    finetgrid = np.exp(np.arange(np.log(0.05), np.log(13.8), 0.01))
    fineivars = np.zeros_like(finetgrid) + np.mean(model.get_lnage_ivars())
    for i in range(16):
        print "parent", i, newpid, "running emcee"
        sampler.run_mcmc(p0, 32)
        p0 = sampler.flatchain[np.argsort(sampler.flatlnprobability)[-nwalkers:]]
        print "parent made new state", p0.shape
        newpid = os.fork()
        if newpid == 0:
            plt.clf()
            plt.plot(xhist, np.log(yhist), "k-")
            for i in range(16):
                plt.plot(model.times, p0[i, :], "bo", alpha=0.5)
                plt.plot(finetgrid,
                         np.log(model.sfr(np.log(finetgrid), p0[i, :], fineivars)),
                         "b-", alpha=0.5)
            plt.plot(model.times, model.prior_mean, "ro", alpha=0.5)
            plt.plot(finetgrid,
                     np.log(model.sfr(np.log(finetgrid), model.prior_mean, fineivars)),
                         "r-", alpha=0.5)
            plt.semilogx()
            plt.xlabel("time ago (Gyr)")
            plt.ylabel("ln (number per ln time)")
            plt.xlim(0.05, 13.8)
            plt.ylim(0., 10.)
            fn = "sfh.png"
            print "child saving", fn
            plt.savefig(fn)
            exit(0)
