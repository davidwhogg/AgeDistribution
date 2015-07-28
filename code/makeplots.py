import numpy as np
import matplotlib.pyplot as plt

def hogg_savefig(fn):
    print "hogg_savefig(): Writing %s ..." % fn
    plt.savefig(fn)
    print "hogg_savefig(): Wrote %s" % fn

if __name__ == "__main__":
    print "Hello World!"

    fn = "../data/HWR_redclump_sample.txt"
    print "Reading %s ..." % fn
    data = np.genfromtxt(fn)
    print data[2]
    print data.shape
    print "Read %s" % fn

    print "Making age histogram ..."
    plt.clf()
    plt.hist(data[:, -1], bins=32, histtype="step")
    plt.xlabel("age (Gyr)")
    hogg_savefig("histogram.png")

    print "Goodbye World!"
