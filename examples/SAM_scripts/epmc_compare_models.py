import numpy as np
import matplotlib.pyplot as plt

Uwxa_full_iwan = np.load('data/Uwxa_full_iwan.npy')
Uwxa_full_bouc = np.load('data/Uwxa_full_bouc.npy')

damps_iwan = Uwxa_full_iwan[:, -2]/2/Uwxa_full_iwan[:, -3]
freqs_iwan = Uwxa_full_iwan[:, -3]/2/np.pi
amps_iwan = Uwxa_full_iwan[:, -1]

damps_bouc = Uwxa_full_bouc[:, -2]/2/Uwxa_full_bouc[:, -3]
freqs_bouc = Uwxa_full_bouc[:, -3]/2/np.pi
amps_bouc = Uwxa_full_bouc[:, -1]


plt.plot(amps_iwan, freqs_iwan, label = 'Iwan')
plt.plot(amps_bouc, freqs_bouc, label = 'Bouc-Wen')
plt.legend()
plt.title("Natural Frequency")
plt.xlabel("Log Modal Amplitude")
plt.ylabel("Natural Frequency")
plt.show()

plt.plot(amps_iwan, damps_iwan, label = 'Iwan')
plt.plot(amps_bouc, damps_bouc, label = 'Bouc-Wen')
plt.legend()
plt.title("Damping Ratio")
plt.xlabel("Log Modal Amplitude")
plt.ylabel("Damping Ratio")
plt.show()

