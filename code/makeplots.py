import numpy as np
import matplotlib.pyplot as plt
import triangle

def hogg_savefig(fn, figure=None):
    print "hogg_savefig(): Writing %s ..." % fn
    if figure is None:
        plt.savefig(fn)
    else:
        figure.savefig(fn)
    print "hogg_savefig(): Wrote %s" % fn

if __name__ == "__main__":
    print "Hello World!"

    fn = "../data/HWR_redclump_sample.txt"
    print "Reading %s ..." % fn
    data = np.genfromtxt(fn)
    names = ["ID", "distance", "Radius_gal", "Phi_gal", "z", "Teff",
             "logg", "[Fe/H]", "[alpha/Fe]", "age"]
    print data[2]
    print data.shape
    print "Read %s" % fn

    print "Making age histogram ..."
    plt.clf()
    plt.hist(data[:, -1], bins=32, histtype="step")
    plt.xlabel("age (Gyr)")
    hogg_savefig("age_hist.png")

    print "Making (R,z) grid..."
    # start magic number time
    Rmin, Rmax = 5., 14. # kpc
    zmin, zmax = -1.5, 1.5 # kpc
    dR = 0.1 # kpc
    young = 4. # Gyr
    alphacut = 0.05 # dex
    # end magic number time
    Rgrid = np.arange(Rmin + 0.5 * dR, Rmax, dR)
    nR = len(Rgrid)
    zgrid = np.arange(zmin + 0.5 * dR, zmax, dR)
    nz = len(zgrid)
    Rgrid = Rgrid[:, None] * (np.ones(nz))[None, :]
    zgrid = (np.ones(nR))[:, None] * zgrid[None, :]
    Nstars = np.zeros_like(Rgrid)
    young_fracs = np.zeros_like(Rgrid) + np.NaN
    low_alpha_fracs = np.zeros_like(Rgrid) + np.NaN
    for rri in range(nR):
        for zzi in range(nz):
            R, z = Rgrid[rri, zzi], zgrid[rri, zzi]
            inside = ((data[:, 2] > R - dR) *
                      (data[:, 2] < R + dR) *
                      (data[:, 4] > z - dR) *
                      (data[:, 4] < z + dR))
            N = np.float(np.sum(inside))
            Nstars[rri, zzi] = N
            if N > 8: # magic 8
                young_fracs[rri, zzi] = np.sum(data[inside, 9] < young) / N
                low_alpha_fracs[rri, zzi] = np.sum(data[inside, 8] < alphacut) / N
    plt.figure(figsize=(12,4))
    plt.clf()
    imshow_kwargs = {"interpolation": "nearest",
                     "aspect": "equal",
                     "origin": "lower",
                     "extent": (Rmin, Rmax, zmin, zmax)}
    plt.imshow(young_fracs.T, cmap=plt.cm.RdBu, vmin=0., vmax=1.,
               **imshow_kwargs)
    plt.xlabel("Galactocentric radius $R$ (kpc)")
    plt.ylabel("Galactic height $z$ (kpc)")
    foo = plt.colorbar()
    foo.set_label("fraction of stars with age < %0.1f Gyr" % young)
    hogg_savefig("young_fracs.png")

    plt.clf()
    plt.imshow(low_alpha_fracs.T, cmap=plt.cm.RdBu, vmin=0., vmax=1.,
               **imshow_kwargs)
    plt.xlabel("Galactocentric radius $R$ (kpc)")
    plt.ylabel("Galactic height $z$ (kpc)")
    foo = plt.colorbar()
    foo.set_label("fraction of stars with %s < %0.2f" % (names[8], alphacut))
    hogg_savefig("low_alpha_fracs.png")

    plt.clf()
    plt.imshow(Nstars.T, cmap=plt.cm.afmhot, vmin=0.,
               **imshow_kwargs)
    plt.xlabel("Galactocentric radius $R$ (kpc)")
    plt.ylabel("Galactic height $z$ (kpc)")
    foo = plt.colorbar()
    foo.set_label(r"number of stars in a $%.1f\times%.1f$ kpc box" %
                  (2. * dR, 2 * dR))
    hogg_savefig("nstars.png")

    print "Making triangle plot ..."
    figure = triangle.corner(data[:, 1:], labels=names[1:])
    hogg_savefig("triangle.png", figure=figure)

    print "Goodbye World!"
