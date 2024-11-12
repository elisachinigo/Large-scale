import numpy as np
from scipy.stats import ks_2samp
from scipy.signal import find_peaks
from scipy.stats import norm
import warnings
from scipy.stats import uniform


def hartigans_dip_test(xpdf):
    """
    Calculates Hartigan's DIP statistic for unimodality.
    
    Parameters:
        xpdf (np.ndarray): A sorted 1D array of sample values.
        
    Returns:
        dip (float): DIP statistic.
        xl (float): Lower bound of the modal interval.
        xu (float): Upper bound of the modal interval.
        ifault (int): Error flag (0 if successful).
        gcm (np.ndarray): Greatest convex minorant.
        lcm (np.ndarray): Least concave majorant.
        mn (np.ndarray): Support indices for gcm.
        mj (np.ndarray): Support indices for lcm.
    """
    
    # Sort x in increasing order and initialize values
    x = np.sort(xpdf)
    N = len(x)
    mn = np.zeros(N, dtype=int)
    mj = np.zeros(N, dtype=int)
    lcm = np.zeros(N, dtype=int)
    gcm = np.zeros(N, dtype=int)
    ifault = 0

    # Check that N is positive (checks for a non empty dataset
    if N <= 0:
        ifault = 1
        print(f"\nHartigansDipTest.    InputError :  ifault={ifault}")
        return 0, 0, 0, ifault, gcm, lcm, mn, mj

    # Check if N is one (single element input)
    if N == 1:
        xl = x[0]
        xu = x[N - 1]
        dip = 0.0
        ifault = 2
        print(f"\nHartigansDipTest.    InputError :  ifault={ifault}")
        return dip, xl, xu, ifault, gcm, lcm, mn, mj

    # Check if all values of X are identical OR if 1 < N < 4 (identical values or too few datapoints)
    if np.all(x == x[0]) or N < 4:
        xl = x[0]
        xu = x[N - 1]
        dip = 0.0
        ifault = 4
        print(f"\nHartigansDipTest.    InputError :  ifault={ifault}")
        return dip, xl, xu, ifault, gcm, lcm, mn, mj

    # Check if X is perfectly unimodal (perfectly unimodal distribution)
    xsign = -np.sign(np.diff(np.diff(x)))
    posi, negi = np.where(xsign > 0)[0], np.where(xsign < 0)[0]
    if len(posi) == 0 or len(negi) == 0 or np.all(posi < min(negi)):
        xl = x[0]
        xu = x[N - 1]
        dip = 0.0
        ifault = 5
        return dip, xl, xu, ifault, gcm, lcm, mn, mj

    # Initialize the convex minorant and concave majorant fits
    fn = float(N)
    low, high = 0, N - 1
    dip = 1.0 / fn
    xl, xu = x[low], x[high]

    # Greatest convex minorant (GCM)
    mn[0] = 0
    for j in range(1, N):
        mn[j] = j - 1
        while mn[j] > 0 and (x[j] - x[mn[j]]) * (mn[j] - mn[mn[j]]) < (x[mn[j]] - x[mn[mn[j]]]) * (j - mn[j]):
            mn[j] = mn[mn[j]]

    # Least concave majorant (LCM)
    mj[N - 1] = N - 1
    for j in range(N - 2, -1, -1):
        mj[j] = j + 1
        while mj[j] < N - 1 and (x[j] - x[mj[j]]) * (mj[j] - mj[mj[j]]) < (x[mj[j]] - x[mj[mj[j]]]) * (j - mj[j]):
            mj[j] = mj[mj[j]]

    iterate = True
    while iterate:
        # Collect change points for GCM and LCM from high to low
        icx, icv = 0, 0
        gcm[icx] = high
        icx += 1
        while gcm[icx - 1] > low:
            gcm[icx] = mn[gcm[icx - 1]]
            icx += 1

        lcm[icv] = low
        icv += 1
        while lcm[icv - 1] < high:
            lcm[icv] = mj[lcm[icv - 1]]
            icv += 1

        # Find the maximum distance greater than 'DIP' between GCM and LCM
        d = 0.0
        for ix in range(icx - 1):
            for iv in range(1, icv):
                if gcm[ix] < lcm[iv]:
                    break
                a = lcm[iv] - gcm[ix - 1]
                b = gcm[ix] - gcm[ix - 1]
                dx = (x[gcm[ix]] - x[gcm[ix - 1]]) * a / (fn * (x[lcm[iv]] - x[gcm[ix - 1]])) - b / fn
                d = max(d, dx)

        iterate = d > dip
        if iterate:
            dip = d

    dip = 0.5 * dip
    xl, xu = x[low], x[high]

    return dip, xl, xu, ifault, gcm[:icx], lcm[:icv], mn, mj
    
def hartigans_dip_signif_test(xpdf, nboot, hartigans_dip_test):
    """
    Calculates Hartigan's DIP statistic and its significance for an empirical PDF.
    
    Parameters:
        xpdf (np.ndarray): Vector of sample values (empirical PDF).
        nboot (int): Number of bootstrap samples.
        hartigans_dip_test (function): Function to calculate the DIP statistic.
        
    Returns:
        dip (float): DIP statistic for xpdf.
        p_value (float): p-value for the DIP statistic.
        xlow (float): Lower bound of the modal interval.
        xup (float): Upper bound of the modal interval.
    """
    # Calculate the DIP statistic from the empirical PDF
    dip, xlow, xup, ifault, gcm, lcm, mn, mj = hartigans_dip_test(xpdf)
    N = len(xpdf)

    # Calculate a bootstrap sample of size NBOOT of the dip statistic for a uniform pdf of sample size N
    boot_dip = []
    for i in range(nboot):
        # Generate a sorted uniform sample
        unifpdfboot = np.sort(uniform.rvs(size=N))
        
        # Calculate the dip statistic for this uniform sample
        unif_dip, _, _, _, _, _, _, _ = hartigans_dip_test(unifpdfboot)
        
        # Append to bootstrapped dip statistics list
        boot_dip.append(unif_dip)
    
    # Sort the bootstrapped dip statistics
    boot_dip = np.sort(boot_dip)
    
    # Calculate p-value: proportion of bootstrap dips greater than or equal to the observed dip
    p_value = np.sum(dip < boot_dip) / nboot

    return dip, p_value, xlow, xup

def bimodal_thresh(bimodaldata, maxthresh=np.inf, Schmidt=True, maxhistbins=25, startbins=10, setthresh=None):
    """
    Calculate the threshold between bimodal data modes (e.g., UP vs DOWN states) and return crossing times.
    
    Args:
    - bimodaldata: A 1D array of bimodal time-series data.
    - maxthresh: Optional, sets a maximum threshold.
    - Schmidt: Optional, uses Schmidt trigger (halfway points between peaks).
    - maxhistbins: Maximum number of histogram bins to try.
    - startbins: Minimum number of histogram bins to start with.
    - setthresh: Manually set the threshold.
    
    Returns:
    - thresh: The calculated threshold between the modes.
    - cross: A dictionary with 'upints' and 'downints' (onsets and offsets of UP/DOWN states).
    - bihist: Bimodal histogram data.
    - diptest: Result from Hartigan's dip test.
    - crossup, crossdown: Crossings for UP and DOWN states.
    """
    
    # Optional parameter handling
    thresh = setthresh if setthresh is not None else None
    
    # Initialize crossings
    crossup, crossdown = np.nan, np.nan
    cross = {'upints': [], 'downints': []}

    # Run Hartigan's dip test for bimodality
    nboot = 500  # number of bootstrap samples
    dip, p_value, _, _ = hartigans_dip_signif_test(bimodaldata, nboot, hartigans_dip_test)
    diptest = {'dip': dip, 'p_value': p_value}

    if p_value > 0.05:  # Not bimodal
        warnings.warn("Dip test indicates data is not bimodal.")
        bihist, _ = np.histogram(bimodaldata, bins=startbins)
        return np.nan, cross, {'hist': bihist, 'bins': startbins}, diptest, crossup, crossdown

    # Remove data above the max threshold
    bimodaldata[bimodaldata >= maxthresh] = np.nan

    numpeaks = 1
    numbins = startbins
    while numpeaks != 2:
        bihist, bin_edges = np.histogram(bimodaldata, bins=numbins)
        peaks, _ = find_peaks(np.concatenate(([0], bihist, [0])), height=None)
        LOCS = peaks - 1  # MATLAB indexing correction
        numbins += 1
        numpeaks = len(LOCS)
        if numbins == maxhistbins and thresh is None:
            warnings.warn("Unable to find bimodal threshold.")
            return np.nan, cross, {'hist': bihist, 'bins': bin_edges}, diptest, crossup, crossdown

    # Define threshold as minimum between peaks
    betweenpeaks = bin_edges[LOCS[0]:LOCS[1]+1]
    dip_vals, _ = find_peaks(-bihist[LOCS[0]:LOCS[1]+1])
    if len(dip_vals) > 0:
        thresh = betweenpeaks[dip_vals[0]]

    # Schmidt trigger: thresholds halfway between trough and peaks
    if Schmidt:
        threshUP = thresh + 0.5 * (betweenpeaks[-1] - thresh)
        threshDOWN = thresh + 0.5 * (betweenpeaks[0] - thresh)
        
        overUP = bimodaldata > threshUP
        overDOWN = bimodaldata > threshDOWN
        
        crossup = np.where(np.diff(overUP.astype(int)) == 1)[0]
        crossdown = np.where(np.diff(overDOWN.astype(int)) == -1)[0]

        # Delete incomplete crossings
        allcrossings = np.vstack((np.column_stack((crossup, np.ones(len(crossup)))),
                                  np.column_stack((crossdown, np.zeros(len(crossdown))))))
        allcrossings = allcrossings[np.argsort(allcrossings[:, 0])]
        updownswitch = np.diff(allcrossings[:, 1])
        allcrossings = np.delete(allcrossings, np.where(updownswitch == 0)[0] + 1, axis=0)

        crossup = allcrossings[allcrossings[:, 1] == 1][:, 0]
        crossdown = allcrossings[allcrossings[:, 1] == 0][:, 0]

        if len(crossup) == 0:
            crossup = np.array([0])
        if len(crossdown) == 0:
            crossdown = np.array([0])
    else:
        overind = bimodaldata > thresh
        crossup = np.where(np.diff(overind.astype(int)) == 1)[0]
        crossdown = np.where(np.diff(overind.astype(int)) == -1)[0]

    # If no crossings were found
    if len(crossup) == 0 or len(crossdown) == 0:
        return np.nan, cross, {'hist': bihist, 'bins': bin_edges}, diptest, crossup, crossdown

    # Define intervals
    upforup = crossup.copy()
    upfordown = crossup.copy()
    downforup = crossdown.copy()
    downfordown = crossdown.copy()

    if crossup[0] < crossdown[0]:
        upfordown = upfordown[1:]
    if crossdown[-1] > crossup[-1]:
        downfordown = downfordown[:-1]
    if crossdown[0] < crossup[0]:
        downforup = downforup[1:]
    if crossup[-1] > crossdown[-1]:
        upforup = upforup[:-1]

    cross['upints'] = np.column_stack((upforup, downforup))
    cross['downints'] = np.column_stack((downfordown, upfordown))

    # Set bihist dictionary with bins and histogram
    bihist = {'hist': bihist, 'bins': bin_edges}

    return thresh, cross, bihist, diptest, crossup, crossdown
