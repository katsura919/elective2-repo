import pandas as pd

# Load CSV
df = pd.read_csv("datasets/pokemon.csv")

# Convert to JSON (key-value pairs of Name -> ID)
pokemon_dict = df.set_index("Name")["#"].to_dict()

# Save as JSON
import json
with open("pokemon.json", "w") as f:
    json.dump(pokemon_dict, f, indent=4)

print("Conversion complete! Saved as pokemon.json")
