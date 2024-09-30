#!/usr/bin/env python3
from csv import DictWriter
import numpy as np

FROM, TO = -40, 70
STEP = 3

np.random.seed(42)

X = np.arange(FROM, TO, STEP, dtype=np.float64)
noise = np.random.normal(0, 0.5, X.shape)
Y = 9/5*X + 32 + noise


with open("dataset.csv", "w") as fp:
    writer = DictWriter(fp, fieldnames=("x", "y"))
    writer.writeheader()
    
    for x, y in zip(X, Y):
        writer.writerow(dict(x=x, y=y))
