import numpy as np
import sys
from collections import Counter, defaultdict

def load_fingerprint(filename):
    """Loads the uncompressed fingerprint from a file."""
    fingerprint = []
    with open(filename, 'r', encoding='latin1') as f:
        for line in f:
            t, x, y, d = map(int, line.strip().split())
            fingerprint.append((t, x, y, d))
    return fingerprint

def fuzzy_match(fp1, fp2, time_tolerance=5, freq_tolerance=5, duration_tolerance=5):
    offset_counter = Counter()

    # Index fp2 by frequency bins for fuzzy lookup
    fp2_freq_index = defaultdict(list)
    for t, x, y, d in fp2:
        fp2_freq_index[x].append((t, y, d))

    for t1, x1, y1, d1 in fp1:
        # Check nearby x bins within freq_tolerance
        x_bins = range(x1 - freq_tolerance, x1 + freq_tolerance + 1)
        for xb in x_bins:
            if xb in fp2_freq_index:
                for t2, y2, d2 in fp2_freq_index[xb]:
                    if abs(y2 - y1) <= freq_tolerance and abs(d2 - d1) <= duration_tolerance:
                        offset = t2 - t1
                        offset_counter[offset] += 1

    if not offset_counter:
        return None, 0

    # Find the most common offset
    best_offset, _ = offset_counter.most_common(1)[0]

    # Count matches around best_offset within time_tolerance
    total_matches = sum(
        count for off, count in offset_counter.items()
        if abs(off - best_offset) <= time_tolerance
    )

    return best_offset, total_matches

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} file1 file2")
    sys.exit(1)

  file1, file2 = sys.argv[1], sys.argv[2]
  fp1 = load_fingerprint(file1)
  fp2 = load_fingerprint(file2)

  offset, matches = fuzzy_match(fp1, fp2)
  print(f"Best match at offset {offset} frames with {matches} matching landmarks.")
