import matplotlib.pyplot as plt
import numpy as np
import pyfluids as pf
from scipy.integrate import cumulative_simpson
from scipy.optimize import brentq

P1 = 120_000.0   #Pa
V1 = 5e-3        #m^3
T = -30.0        #grader celsius
V_min = 0.1e-3   #m^3 

#skapar ett luft objekt
air = pf.Fluid(pf.FluidsList.Air)

#Beräkna sepc v och u i start
air.update(pf.Input.pressure(P1), pf.Input.temperature(T))
v1 = air.specific_volume  
u1 = air.internal_energy  
m = V1 / v1             


#förberadelse arr för plottning
V = np.geomspace(V1, V_min, 1001)
P_ideal = P1 * V1 / V
P_verklig = np.empty_like(V)
u_verklig = np.empty_like(V)
P_verklig[0] = P1
u_verklig[0] = u1

#aviklse från önksad temperatur
def temperaturavvikelse(P, v):
    #Beräkna temp för P och v
    air.update(pf.Input.pressure(P), pf.Input.specific_volume(v))
    return air.temperature - T

"looopa volymer, berääkna P och u för alla volymer"
for i in range(1, len(V)):
    v = V[i] / m

    #Finn P som res. i temp. avvikelse = 0
    P = brentq(
        temperaturavvikelse,
        0.8 * P_ideal[i],
        1.2 * P_ideal[i],
        args=(v,),
    )
    P_verklig[i] = P

    #Beräkna u för P oh T
    air.update(pf.Input.pressure(P), pf.Input.temperature(T))
    u_verklig[i] = air.internal_energy

# Beräkna abs W och absQ för ideal gas
x = np.log(V1 / V)
W_ideal = P1 * V1 * x
Q_ideal = W_ideal
#beräkna W och Q för verklig gas
W_verklig = cumulative_simpson(P_verklig * V, x=x, initial=0.0)
Q_verklig = W_verklig - m * (u_verklig - u1)

#Plotta res 
V_dm3 = V * 1e3
fig, (ax, ax_diff) = plt.subplots(
    2, 1, figsize=(8, 7), sharex=True, layout="constrained"
)

ax.plot(V_dm3, W_ideal, "k:", label="Ideal gas – arbete och värme")
ax.plot(V_dm3, W_verklig, label="PyFluids – tillfört arbete")
ax.plot(V_dm3, Q_verklig, "--", label="PyFluids – bortförd värme")
ax.set_title("Isoterm kompression av luft vid −30 °C")
ax.set_ylabel("Energi [J]")

ax_diff.plot(V_dm3, W_verklig - W_ideal, label="Tillfört arbete")
ax_diff.plot(V_dm3, Q_verklig - Q_ideal, "--", label="Bortförd värme")
ax_diff.axhline(0, color="0.4", linewidth=0.8)
ax_diff.set_ylabel("PyFluids − ideal gas [J]")
ax_diff.set_xlabel("Slutvolym [dm³]")
ax_diff.set_xscale("log")
ax_diff.set_xticks([0.1, 0.2, 0.5, 1, 2, 5], ["0,1", "0,2", "0,5", "1", "2", "5"])

for subplot in (ax, ax_diff):
    subplot.grid(alpha=0.25)
    subplot.legend()

fig.savefig("isoterm_kompression.png", dpi=180)
plt.show()