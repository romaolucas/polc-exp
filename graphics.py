import numpy as np
import matplotlib.pyplot as plt
from extractor import bow

corpus, labels = bow.get_info_from("output_clam.csv")
classes = set(labels)

freq = {}
for c in classes:
    freq[c] = 0
    for lbl in labels:
        if c == lbl:
            freq[c] += 1

x = np.arange(len(classes))
print(freq)
frequencies = list(freq.values())
print(frequencies)
fig, ax = plt.subplots()
plt.bar(x, frequencies)
plt.xticks(x, classes)
plt.savefig("fig_classes.png")
