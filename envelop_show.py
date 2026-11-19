import numpy as np
import matplotlib.pyplot as plt

envelope_stacked = np.load(f"./results/envelope_stacked.npy")
print(envelope_stacked.shape)
envelope_max = np.max(envelope_stacked, axis=1)
envelope_normalized = envelope_stacked / envelope_max[:, np.newaxis]

envelope_avg = np.mean(envelope_normalized, axis=0)
envelope_std = np.std(envelope_normalized, axis=0)

envelope_avg_normalized = envelope_avg / np.max(envelope_avg)
np.save(f"./results/envelope.npy", envelope_avg_normalized)

fig, axs = plt.subplots(figsize=(10, 5))
x = np.arange(envelope_avg.shape[0])
axs.plot(x, envelope_avg, color="C0", label="mean")
axs.fill_between(
    x,
    envelope_avg - envelope_std,
    envelope_avg + envelope_std,
    color="C0",
    alpha=0.3,
    linewidth=0,
    label="mean ± std",
)
axs.legend()
plt.show()
