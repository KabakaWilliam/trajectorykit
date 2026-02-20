from trajectorykit import dispatch

EXAMPLE_1 = "Compare the stats of Blue Eyes White Dragon vs Dark Magician vs TimeWizard Yu-Gi-Oh cards and visualise them. Put a star on the strongest monster. Next look up a spell card that when applied to each card, would make it the strongest. Do not make this up please. visualise this monster + card combination."

EXAMPLE_2 = "What is the general consensus on George RR Martin releasing the Winds of Winter this year? If it's not this year, then what year is the most likely then according to what you find out there."

EXAMPLE_3 = "Compare the 2022-23 season stats of Erling Haaland, Harry Kane, and Kylian Mbappé (goals, assists, shots per game).Visualise the comparison. Put a crown icon on the most efficient scorer (goals per shot). Then find a real tactical formation change their club used that improved attacking output and show how each player would perform under that system. Do not fabricate tactical data."

EXAMPLE_4 = "Compare the base stats of Charizard, Blastoise, and Venusaur. Visualise HP, Attack, Defense, Special Attack, Special Defense, and Speed. Mark the strongest overall stat total. Then identify a real held item from the Pokémon games that would maximize each Pokémon's strongest stat. Show the new adjusted stat profile."

EXAMPLE_5 = "Compare the canonical power scaling or feats of Goku, Naruto Uzumaki, and Ichigo Kurosaki at their strongest confirmed forms. Visualise destructive capability tiers. Mark the strongest based on feats only. Then apply a canonical power multiplier from their series (e.g., form transformation) and re-rank then visualise."

result = dispatch(
    user_input=EXAMPLE_4,
    turn_length=None,
    max_tokens=32768,
    verbose=True
)

print("\n📜 Full Trace Tree:")
result["trace"].pretty_print()

# Save trace to disk
path = result["trace"].save()
print(f"\n💾 Trace saved to: {path}")