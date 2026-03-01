import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('CeO2_5s_71p676keV_3300mm_0deg_002011.ge1.h5.corr.csv', skip_header=1)
data = data[data[:, 7] == 0]  # filter outliers
data[data[:, 3] < 100, 3] += 360
print(data.shape)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(data[:, 3], data[:, 15], c=data[:, 10], cmap='tab10', label='Fitted Peaks')
plt.colorbar(scatter, label='Ring Number')

ideal_a = data[0, 14]
plt.axhline(ideal_a, color='red', linestyle='-', alpha=0.5, label=f'Ideal a ({ideal_a:.6f} Å)')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlabel('Eta (deg)')
plt.ylabel('Lattice Parameter a (Å)')
plt.title('CalibrantPanelShiftsOMP Output')
plt.show()
