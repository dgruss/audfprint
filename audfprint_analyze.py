import os
import numpy as np

import scipy.signal

import audio_read
import stft
import sys

def locmax(vec, indices=False):
    """ Return a boolean vector of which points in vec are local maxima.
        End points are peaks if larger than single neighbors.
        if indices=True, return the indices of the True values instead
        of the boolean vector.
    """
    # vec[-1]-1 means last value can be a peak
    # nbr = np.greater_equal(np.r_[vec, vec[-1]-1], np.r_[vec[0], vec])
    # the np.r_ was killing us, so try an optimization...
    nbr = np.zeros(len(vec) + 1, dtype=bool)
    nbr[0] = True
    nbr[1:-1] = np.greater_equal(vec[1:], vec[:-1])
    maxmask = (nbr[:-1] & ~nbr[1:])
    if indices:
        return np.nonzero(maxmask)[0]
    else:
        return maxmask


OVERSAMP = 1
N_FFT = 512
N_HOP = 32
HPF_POLE = 0.98
sp_width = None
sp_len = None
sp_vals = []

density = 1
target_sr = 11025
n_fft = N_FFT
n_hop = N_HOP
# how wide to spreak peaks
f_sd = 30.0
# Maximum number of local maxima to keep per frame
maxpksperframe = 5
# Limit the num of pairs we'll make from each peak (Fanout)
maxpairsperpeak = 3
# Values controlling peaks2landmarks
# +/- 31 bins in freq (LIMITED TO -32..31 IN LANDMARK2HASH)
targetdf = 31
# min time separation (traditionally 1, upped 2014-08-04)
mindt = 2
# max lookahead in time (LIMITED TO <64 IN LANDMARK2HASH)
targetdt = 63
# global stores duration of most recently-read soundfile
soundfiledur = 0.0
# .. and total amount of sound processed
soundfiletotaldur = 0.0
# .. and count of files
soundfilecount = 0
# Control behavior on file reading error
fail_on_error = True

def spreadpeaksinvector(vector, width=4.0):
    """ Create a blurred version of vector, where each of the local maxes
        is spread by a gaussian with SD <width>.
    """
    npts = len(vector)
    peaks = locmax(vector, indices=True)
    return spreadpeaks(zip(peaks, vector[peaks]),
                            npoints=npts, width=width)

def spreadpeaks(peaks, npoints=None, width=4.0, base=None):
    """ Generate a vector consisting of the max of a set of Gaussian bumps
    :params:
      peaks : list
        list of (index, value) pairs giving the center point and height
        of each gaussian
      npoints : int
        the length of the output vector (needed if base not provided)
      width : float
        the half-width of the Gaussians to lay down at each point
      base : np.array
        optional initial lower bound to place Gaussians above
    :returns:
      vector : np.array(npoints)
        the maximum across all the scaled Gaussians
    """
    global sp_width, sp_len, sp_vals
    if base is None:
        vec = np.zeros(npoints)
    else:
        npoints = len(base)
        vec = np.copy(base)
    # binvals = np.arange(len(vec))
    # for pos, val in peaks:
    #   vec = np.maximum(vec, val*np.exp(-0.5*(((binvals - pos)
    #                                /float(width))**2)))
    if width != sp_width or npoints != sp_len:
        # Need to calculate new vector
        sp_width = width
        sp_len = npoints
        sp_vals = np.exp(-0.5 * ((np.arange(-npoints, npoints + 1)
                                          / width)**2))
    # Now the actual function
    for pos, val in peaks:
        vec = np.maximum(vec, val * sp_vals[np.arange(npoints)
                                                    + npoints - pos])
    return vec

def _decaying_threshold_fwd_prune(sgram, a_dec):
    """ forward pass of findpeaks
        initial threshold envelope based on peaks in first 10 frames
    """
    (srows, scols) = np.shape(sgram)
    sthresh = spreadpeaksinvector(
        np.max(sgram[:, :np.minimum(10, scols)], axis=1), f_sd
    )
    # Store sthresh at each column, for debug
    # thr = np.zeros((srows, scols))
    peaks = np.zeros((srows, scols))
    # optimization of mask update
    sp_pts = len(sthresh)
    sp_v = sp_vals

    for col in range(scols):
        s_col = sgram[:, col]
        # Find local magnitude peaks that are above threshold
        sdmaxposs = np.nonzero(locmax(s_col) * (s_col > sthresh))[0]
        # Work down list of peaks in order of their absolute value
        # above threshold
        valspeaks = sorted(zip(s_col[sdmaxposs], sdmaxposs), reverse=True)
        for val, peakpos in valspeaks[:maxpksperframe]:
            # What we actually want
            # sthresh = spreadpeaks([(peakpos, s_col[peakpos])],
            #                      base=sthresh, width=f_sd)
            # Optimization - inline the core function within spreadpeaks
            sthresh = np.maximum(sthresh,
                                  val * sp_v[(sp_pts - peakpos):
                                              (2 * sp_pts - peakpos)])
            peaks[peakpos, col] = 1
        sthresh *= a_dec
    return peaks

def _decaying_threshold_bwd_prune_peaks(sgram, peaks, a_dec):
    """ backwards pass of findpeaks """
    scols = np.shape(sgram)[1]
    # Backwards filter to prune peaks
    sthresh = spreadpeaksinvector(sgram[:, -1], f_sd)
    for col in range(scols, 0, -1):
        pkposs = np.nonzero(peaks[:, col - 1])[0]
        peakvals = sgram[pkposs, col - 1]
        for val, peakpos in sorted(zip(peakvals, pkposs), reverse=True):
            if val >= sthresh[peakpos]:
                # Setup the threshold
                sthresh = spreadpeaks([(peakpos, val)], base=sthresh,
                                            width=f_sd)
                # Delete any following peak (threshold should, but be sure)
                if col < scols:
                    peaks[peakpos, col] = 0
            else:
                # delete the peak
                peaks[peakpos, col - 1] = 0
        sthresh = a_dec * sthresh
    return peaks

def find_peaks(d, sr):
    """ Find the local peaks in the spectrogram as basis for fingerprints.
        Returns a list of (time_frame, freq_bin) pairs.

    :params:
      d - np.array of float
        Input waveform as 1D vector

      sr - int
        Sampling rate of d (not used)

    :returns:
      pklist - list of (int, int)
        Ordered list of landmark peaks found in STFT.  First value of
        each pair is the time index (in STFT frames, i.e., units of
        n_hop/sr secs), second is the FFT bin (in units of sr/n_fft
        Hz).
    """
    if len(d) == 0:
        return []

    # masking envelope decay constant
    a_dec = (1 - 0.01 * (density * np.sqrt(n_hop / 352.8) / 35)) ** (1 / OVERSAMP)
    # Take spectrogram
    mywin = np.hanning(n_fft + 2)[1:-1]
    sgram = np.abs(stft.stft(d, n_fft=n_fft,
                              hop_length=n_hop,
                              window=mywin))
    sgrammax = np.max(sgram)
    if sgrammax > 0.0:
        sgram = np.log(np.maximum(sgram, np.max(sgram) / 1e6))
        sgram = sgram - np.mean(sgram)
    else:
        # The sgram is identically zero, i.e., the input signal was identically
        # zero.  Not good, but let's let it through for now.
        print("find_peaks: Warning: input signal is identically zero.")
    # High-pass filter onset emphasis
    # [:-1,] discards top bin (nyquist) of sgram so bins fit in 8 bits
    sgram = np.array([scipy.signal.lfilter([1, -1],
                                            [1, -HPF_POLE ** (1 / OVERSAMP)], s_row)
                      for s_row in sgram])[:-1, ]
    # Prune to keep only local maxima in spectrum that appear above an online,
    # decaying threshold
    peaks = _decaying_threshold_fwd_prune(sgram, a_dec)
    # Further prune these peaks working backwards in time, to remove small peaks
    # that are closely followed by a large peak
    peaks = _decaying_threshold_bwd_prune_peaks(sgram, peaks, a_dec)
    # build a list of peaks we ended up with
    scols = np.shape(sgram)[1]
    pklist = []
    for col in range(scols):
        for bin_ in np.nonzero(peaks[:, col])[0]:
            pklist.append((col, bin_))
    return pklist

def peaks2landmarks(pklist):
    """ Take a list of local peaks in spectrogram
        and form them into pairs as landmarks.
        pklist is a column-sorted list of (col, bin) pairs as created
        by findpeaks().
        Return a list of (col, peak, peak2, col2-col) landmark descriptors.
    """
    # Form pairs of peaks into landmarks
    landmarks = []
    if len(pklist) > 0:
        # Find column of the final peak in the list
        scols = pklist[-1][0] + 1
        # Convert (col, bin) list into peaks_at[col] lists
        peaks_at = [[] for _ in range(scols)]
        for (col, bin_) in pklist:
            peaks_at[col].append(bin_)

        # Build list of landmarks <starttime F1 endtime F2>
        for col in range(scols):
            for peak in peaks_at[col]:
                pairsthispeak = 0
                for col2 in range(col + mindt,
                                    min(scols, col + targetdt)):
                    if pairsthispeak < maxpairsperpeak:
                        for peak2 in peaks_at[col2]:
                            if abs(peak2 - peak) < targetdf:
                                # and abs(peak2-peak) + abs(col2-col) > 2 ):
                                if pairsthispeak < maxpairsperpeak:
                                    # We have a pair!
                                    landmarks.append((col, peak,
                                                      peak2, col2 - col))
                                    pairsthispeak += 1

    return landmarks

def wavfile2peaks(filename):
    global soundfiledur, soundfiletotaldur, soundfilecount
    try:
        # [d, sr] = librosa.load(filename, sr=target_sr)
        d, sr = audio_read.audio_read(filename, sr=target_sr, channels=1)
    except Exception as e:  # audioread.NoBackendError:
        message = "wavfile2peaks: Error reading " + filename
        if fail_on_error:
            print(e)
            raise IOError(message)
        print(message, "skipping")
        d = []
        sr = target_sr
    # Store duration in a global because it's hard to handle
    dur = len(d) / sr
    # Calculate hashes with optional part-frame shifts
    peaks = find_peaks(d, sr)

    # instrumentation to track total amount of sound processed
    soundfiledur = dur
    soundfiletotaldur += dur
    soundfilecount += 1
    return peaks

def wavfile2hashes(filename):
    peaks = wavfile2peaks(filename)
    if len(peaks) == 0:
        return []
    query_hashes = peaks2landmarks(peaks)

    np.unique(query_hashes, axis=0)

    return query_hashes

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python audfprint_analyze.py <audio_file> <fingerprint_file>")
    sys.exit(1)
  test_fn = sys.argv[1]
  hashes = wavfile2hashes(test_fn)
  print(f"N_HOP: {n_hop}, N_HOP/SR: {n_hop / target_sr * 1000} ms per frame")
  with open(sys.argv[2], 'w') as f:
    for t, x, y, d in hashes:
        f.write(f"{t} {x} {y} {d}\n")
