import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load Pokémon dataset
pokemon = pd.read_csv("datasets/pokemon.csv")

# Create a dictionary mapping Pokémon ID to their stats and types
pokemon_dict = pokemon.set_index("#").to_dict(orient="index")

# Pokémon type effectiveness matrix
type_chart = {
    "Normal": {"Rock": 0.5, "Ghost": 0, "Steel": 0.5},
    "Fire": {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 2, "Bug": 2, "Rock": 0.5, "Dragon": 0.5, "Steel": 2},
    "Water": {"Fire": 2, "Water": 0.5, "Grass": 0.5, "Ground": 2, "Rock": 2, "Dragon": 0.5},
    "Electric": {"Water": 2, "Electric": 0.5, "Grass": 0.5, "Ground": 0, "Flying": 2, "Dragon": 0.5},
    "Grass": {"Fire": 0.5, "Water": 2, "Grass": 0.5, "Poison": 0.5, "Ground": 2, "Flying": 0.5, "Bug": 0.5, "Rock": 2, "Dragon": 0.5, "Steel": 0.5},
}

# Function to calculate type effectiveness
def get_type_effectiveness(attacker_type1, attacker_type2, defender_type1, defender_type2):
    effectiveness = 1.0
    for atk_type in [attacker_type1, attacker_type2]:
        if pd.isna(atk_type):
            continue
        for def_type in [defender_type1, defender_type2]:
            if pd.isna(def_type):
                continue
            effectiveness *= type_chart.get(atk_type, {}).get(def_type, 1.0)
    return effectiveness

# Feature extraction function
def get_features(first_pokemon, second_pokemon):
    p1 = pokemon_dict.get(first_pokemon)
    p2 = pokemon_dict.get(second_pokemon)
    
    if not p1 or not p2:
        print("Invalid Pokémon ID. Please try again.")
        return None
    
    type_effectiveness = get_type_effectiveness(p1["Type 1"], p1["Type 2"], p2["Type 1"], p2["Type 2"])
    return [
        p1["HP"], p1["Attack"], p1["Defense"], p1["Sp. Atk"], p1["Sp. Def"], p1["Speed"], int(p1["Legendary"]),
        p2["HP"], p2["Attack"], p2["Defense"], p2["Sp. Atk"], p2["Sp. Def"], p2["Speed"], int(p2["Legendary"]),
        type_effectiveness
    ], p1["Name"], p2["Name"]

# Load trained model
model = keras.models.load_model("pokemon_battle_model.h5")

# User input for Pokémon battle
first_pokemon = int(input("Enter First Pokémon ID: "))
second_pokemon = int(input("Enter Second Pokémon ID: "))

features, first_pokemon_name, second_pokemon_name = get_features(first_pokemon, second_pokemon)

if features:
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    winner = first_pokemon if prediction[0][0] > 0.5 else second_pokemon
    winner_name = first_pokemon_name if prediction[0][0] > 0.5 else second_pokemon_name
    print(f"Predicted Winner: {winner_name} (Pokémon ID {winner})")
