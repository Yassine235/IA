import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ville = ["Ndjamena", "Moundou", "Sarh", "Abeche"]
population = [5.12, 2.98, 2.1, 3.9]

df = pd.DataFrame({"ville" : ville,
                   "population" : population})
print(f"Voici les differentes informations: \n {df}")

plt.plot(ville, population, label="Popoulation")
plt.legend()
plt.title("Graphic representation  about population")
plt.xlabel("ville")
plt.ylabel("Population")
plt.grid(True)
plt.show()