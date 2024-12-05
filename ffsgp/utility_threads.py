# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

import os


def get_nthreads(n_jobs: int) -> int:
    """
    Compute n_jobs following scikit-learn conventions.

    n_jobs>0: set maximum number of concurrently running workers
    n_jobs<0: (n_cpus + 1 + n_jobs) are used. Example n_jobs=-2: all CPUs but one are used.
    n_jobs=0: ERROR
    """
    total_cores = os.cpu_count()
    if total_cores is None:
        raise RuntimeError("Unable to determine the number of CPU cores.")
    
    if n_jobs > 0:
        if n_jobs <= total_cores:
            return n_jobs
        else:
            return total_cores
    elif n_jobs < 0:
        # Calculate cores to use based on the negative value
        threads = total_cores + n_jobs + 1
        if threads > 0:
            return threads
        else:
            raise ValueError(f"n_jobs={n_jobs} results in a non-positive number of threads ({threads}).")
    else:
        # Raise an error for invalid n_jobs=0 or other unexpected values
        raise ValueError(f"Invalid value for n_jobs: {n_jobs}. Must be a different from 0.")
