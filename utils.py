import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.ndimage import gaussian_filter

from scipy.special import erf

from lmfit import Model


def gausssphere_profile(z=np.linspace(0, 2, 100), Z=0.8):
    """Radial profile of a sphere convolved with a 3D radial symmetric Gaussian

     Parameters
     ----------
     z : 1D numpy float array
       normalized radial coordinate    (r / (sqrt(2) * sigma))

     Z : float
       normalized radius of the sphere (R / (sqrt(2) * sigma))

    Returns
    -------
    1D numpy array
    """

    sqrtpi = np.sqrt(np.pi)

    P = np.zeros_like(z)

    inds0 = np.argwhere(z == 0)
    inds1 = np.argwhere(z != 0)

    P[inds0] = erf(Z) - 2 * Z * np.exp(-(Z**2)) / sqrtpi

    P[inds1] = 0.5 * (erf(z[inds1] + Z) - erf(z[inds1] - Z)) - (0.5 / sqrtpi) * (
        (np.exp(-((z[inds1] - Z) ** 2)) - np.exp(-((z[inds1] + Z) ** 2))) / z[inds1]
    )

    return P


# --------------------------------------------------------------------------------------------------


def glasssphere_profile(r, R=18.5, FWHM=5, d=1.5, S=10.0, B=1.0):
    """Radial profile of a hot sphere with cold glass wall in warm background

    Parameters
    ----------
    r : 1D numpy float array
      array with radial coordinates

    R : float, optional
      the radius of the sphere

    FWHM : float, optional
      the full width at half maximum of the points spread function

    d : float, optional
      the thickness (diameter) of the cold glass wall

    S : float, optional
      the signal in the sphere

    B : float, optional
      the signal in the background

    Returns
    -------
    1D numpy float array
    """
    sqrt2 = np.sqrt(2)

    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    Z = R / (sigma * sqrt2)
    w = d / (sigma * sqrt2)
    z = r / (sigma * sqrt2)

    P = S * gausssphere_profile(z, Z) - B * gausssphere_profile(z, Z + w) + B

    return P


def get_sphere_center(vol, voxsizes, relth=0.25):
    """Get the center of gravity of a single hot sphere in a volume

    Parameters
    ----------
    vol : 3d numpy array
      containing the volume

    voxsizes : 3 component array
      with the voxel sizes

    relth : float, optional
      the relative threshold (signal over background) for the first coarse
      delination of the sphere - default 0.25
    """
    # now we have to find the activity weighted center of gravity of the sphere
    # to do so we do a coarse delineation of the sphere (30% over bg)
    bg = np.mean(vol[:, :, 0])
    absth = relth * (vol.max() - bg) + bg

    mask = np.zeros_like(vol, dtype=np.uint8)
    mask[vol > absth] = 1

    i0, i1, i2 = np.indices(vol.shape)
    i0 = i0 * voxsizes[0]
    i1 = i1 * voxsizes[1]
    i2 = i2 * voxsizes[2]

    # calculate the maxmimum radius of the subvolumes
    # all voxels with a distance bigger than rmax will not be included in the fit
    rmax = np.min((i0.max(), i1.max(), i2.max())) / 2

    # first try to get the center of mass via the coarse delineation
    weights = vol[mask == 1]
    summedweights = np.sum(weights)

    c0 = np.sum(i0[mask == 1] * weights) / summedweights
    c1 = np.sum(i1[mask == 1] * weights) / summedweights
    c2 = np.sum(i2[mask == 1] * weights) / summedweights

    r = np.sqrt((i0 - c0) ** 2 + (i1 - c1) ** 2 + (i2 - c2) ** 2)

    # second try to get the center of mass
    # use weights from a smoothed volume
    sigmas = 4 / (2.355 * voxsizes)
    vol_sm = gaussian_filter(vol, sigma=sigmas)

    weights = vol_sm[r <= rmax]
    summedweights = np.sum(weights)

    d0 = np.sum(i0[r <= rmax] * weights) / summedweights
    d1 = np.sum(i1[r <= rmax] * weights) / summedweights
    d2 = np.sum(i2[r <= rmax] * weights) / summedweights

    sphere_center = np.array([d0, d1, d2])

    return sphere_center


# --------------------------------------------------------------------------------------------------


def fitspheresubvolume(
    vol,
    voxsizes,
    relth=0.25,
    Rfix=None,
    FWHMfix=None,
    dfix=None,
    Sfix=None,
    Bfix=None,
    wm="dist",
    cl=False,
    sphere_center=None,
):
    """Fit the radial sphere profile of a 3d volume containg 1 sphere

    Parameters
    ----------
    vol : 3d numpy array
      containing the volume

    voxsizes : 3 component array
      with the voxel sizes

    relth : float, optional
      the relative threshold (signal over background) for the first coarse
      delination of the sphere

    dfix, Sfix, Bfix, Rfix : float, optional
      fixed values for the wall thickness, signal, background and radius

    wm : string, optinal
      the weighting method of the data (equal, dist, sqdist)

    cl : bool, optional
      bool whether to compute the confidence limits (this takes very long)

    sphere_center : 3 element np.array
      containing the center of the spheres in mm
      this is the center of in voxel coordiantes multiplied by the voxel sizes

    Returns
    -------
    Dictionary
      with the fitresults (as returned by lmfit)
    """

    if sphere_center is None:
        sphere_center = get_sphere_center(vol, voxsizes, relth=relth)

    i0, i1, i2 = np.indices(vol.shape)
    i0 = i0 * voxsizes[0]
    i1 = i1 * voxsizes[1]
    i2 = i2 * voxsizes[2]

    rmax = np.min((i0.max(), i1.max(), i2.max())) / 2
    r = np.sqrt(
        (i0 - sphere_center[0]) ** 2
        + (i1 - sphere_center[1]) ** 2
        + (i2 - sphere_center[2]) ** 2
    )

    data = vol[r <= rmax].flatten()
    rfit = r[r <= rmax].flatten()

    if Rfix == None:
        Rinit = 0.5 * rmax
    else:
        Rinit = Rfix

    if FWHMfix == None:
        FWHMinit = 2 * voxsizes[0]
    else:
        FWHMinit = FWHMfix

    if dfix == None:
        dinit = 0.15
    else:
        dinit = dfix

    if Sfix == None:
        Sinit = data.max()
    else:
        Sinit = Sfix

    if Bfix == None:
        Binit = data.min()
    else:
        Binit = Bfix

    # lets do the actual fit
    pmodel = Model(glasssphere_profile)
    params = pmodel.make_params(R=Rinit, FWHM=FWHMinit, d=dinit, S=Sinit, B=Binit)

    # fix the parameters that should be fixed
    if Rfix != None:
        params["R"].vary = False
    if FWHMfix != None:
        params["FWHM"].vary = False
    if dfix != None:
        params["d"].vary = False
    if Sfix != None:
        params["S"].vary = False
    if Bfix != None:
        params["B"].vary = False

    params["R"].min = 0
    params["FWHM"].min = 0
    params["d"].min = 0
    params["S"].min = 0
    params["B"].min = 0

    if wm == "equal":
        weights = np.ones_like(rfit)
    elif wm == "sqdist":
        weights = 1.0 / (rfit**2)
    else:
        weights = 1.0 / rfit

    weights[weights == np.inf] = 0

    fitres = pmodel.fit(data, r=rfit, params=params, weights=weights)
    fitres.rdata = rfit
    if cl:
        fitres.cls = fitres.conf_interval()

    # calculate the a50 mean
    fitres.a50th = fitres.values["B"] + 0.5 * (vol.max() - fitres.values["B"])
    fitres.mean_a50 = np.mean(data[data >= fitres.a50th])

    # calculate the mean
    fitres.mean = np.mean(data[rfit <= fitres.values["R"]])

    # calculate the max
    fitres.max = data.max()

    # add the sphere center to the fit results
    fitres.sphere_center = sphere_center

    return fitres


# --------------------------------------------------------------------------------------------------


def plotspherefit(fitres, ax=None, xlim=None, ylim=None, unit="mm", showres=True):
    """Plot the results of a single sphere fit

    Parameters
    ----------
    fitres : dictionary
      the results of the fit as returned by fitspheresubvolume

    ax : matplotlib axis, optional
      to be used for the plot

    xlim, ylim : float, optional
      the x/y limit

    unit : str, optional
      the unit of the radial coordinate

    showres : bool, optional
      whether to add text about the fit results in the plot
    """

    rplot = np.linspace(0, fitres.rdata.max(), 100)

    if ax == None:
        fig, ax = plt.subplots(1)
    if xlim == None:
        xlim = (0, rplot.max())

    ax.plot(fitres.rdata, fitres.data, "k.", ms=2.5)

    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            fitres.values["R"],
            fitres.values["S"],
            facecolor="lightgrey",
            edgecolor="None",
        )
    )
    x2 = fitres.values["R"] + fitres.values["d"]
    dx2 = xlim[1] - x2
    ax.add_patch(
        patches.Rectangle(
            (x2, 0), dx2, fitres.values["B"], facecolor="lightgrey", edgecolor="None"
        )
    )

    ax.plot(rplot, fitres.eval(r=rplot), "r-")
    ax.set_xlabel("R (" + unit + ")")
    ax.set_ylabel("signal")

    ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)

    if showres:
        ax.text(
            0.99,
            0.99,
            fitres.fit_report(),
            fontsize=6,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
        )
