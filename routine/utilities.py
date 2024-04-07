def normalize(a):
    amax, amin = a.max(), a.min()
    diff = amax - amin
    if diff > 0:
        return (a - amin) / diff
    else:
        return a
