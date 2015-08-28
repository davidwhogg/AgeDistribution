"""
This file is part of the AgeDistribution project.
Copyright 2015 David W. Hogg (NYU).

Take the output of the cross-validation and infer the variance of the
estimator.

## bugs:
- Does not yet optimize!
"""

import numpy as np
import matplotlib.pyplot as plt
log2pi = np.log(2. * np.pi)

def ln_1d_gaussian(x, mu, var):
    return -0.5 * log2pi - 0.5 * (x - mu) ** 2 / var - 0.5 * np.log(var)

def ln_like(pars, ln_masses_Cannon, dln_masses):
    m, b = pars
    vars = (m * ln_masses_Cannon + b) ** 2
    return np.sum(ln_1d_gaussian(dln_masses, 0., vars))

if __name__ == "__main__":
    ln_masses_Kepler, ln_masses_Cannon = \
        np.genfromtxt("../data/ln_mass_input_output_crossvalidation.txt",
                      usecols=(0,1), unpack=1, skip_header=1)
    good = (ln_masses_Cannon > -0.3) * (ln_masses_Cannon < 1.0)
    ln_masses_Cannon = ln_masses_Cannon[good]
    ln_masses_Kepler = ln_masses_Kepler[good]
    dln_masses = ln_masses_Cannon - ln_masses_Kepler

    m0 = 0.082
    b0 = 0.132
    pars = np.array([m0, b0])
    step = 0.001
    bs = np.arange(0.05 + 0.5 * step, 0.25, step)
    ln_likes = np.zeros_like(bs)
    for ii,b in enumerate(bs):
        pars[1] = b
        ln_likes[ii] = ln_like(pars, ln_masses_Cannon, dln_masses)
    plt.clf()
    plt.plot(bs, ln_likes, "k-")
    plt.axvline(b0, color="k", alpha=0.5)
    plt.xlabel("b")
    plt.ylabel("ln likelihood")
    plt.title("m = %.3f" % m0)
    plt.savefig("infer_variance_2.png")

    pars = np.array([m0, b0])
    step = 0.001
    ms = np.arange(0.0 + 0.5 * step, 0.2, step)
    ln_likes = np.zeros_like(ms)
    for ii,m in enumerate(ms):
        pars[0] = m
        ln_likes[ii] = ln_like(pars, ln_masses_Cannon, dln_masses)
    plt.clf()
    plt.plot(ms, ln_likes, "k-")
    plt.axvline(m0, color="k", alpha=0.5)
    plt.xlabel("m")
    plt.ylabel("ln likelihood")
    plt.title("b = %.3f" % b0)
    plt.savefig("infer_variance_3.png")

    plt.clf()
    plt.plot(ln_masses_Cannon, dln_masses, "k.")
    plt.xlabel("ln(M) from Cannon")
    plt.ylabel("ln(M) difference (Cannon - Kepler)")
    xlim = np.array(plt.xlim())
    sigma = m0 * xlim + b0
    plt.plot(xlim, sigma, "k-", alpha=0.5)
    plt.plot(xlim, -sigma, "k-", alpha=0.5)
    plt.savefig("infer_variance_1.png")

