from trajectorykit import dispatch

EXAMPLE_1 = "Compare the stats of Blue Eyes White Dragon vs Dark Magician vs TimeWizard Yu-Gi-Oh cards and visualise them. Put a star on the strongest monster. Next look up a spell card that when applied to each card, would make it the strongest. This card can be different per monster. Do not make this up please. visualise this monster + spell card combination indicating which card was used."

EXAMPLE_2 = "What is the general consensus on George RR Martin releasing the Winds of Winter this year? If it's not this year, then what year is the most likely then according to what you find out there."

EXAMPLE_3 = "Compare the 2022-23 season stats of Erling Haaland, Harry Kane, and Kylian Mbappé (goals, assists, shots per game).Visualise the comparison. Put a crown icon on the most efficient scorer (goals per shot). Then find a real tactical formation change their club used that improved attacking output and show how each player would perform under that system. Do not fabricate tactical data."

EXAMPLE_4 = "Compare the base stats of Charizard, Blastoise, and Venusaur. Visualise HP, Attack, Defense, Special Attack, Special Defense, and Speed. Mark the strongest overall stat total. Then identify a real held item from the Pokémon games that would maximize each Pokémon's strongest stat. Show the new adjusted stat profile."

EXAMPLE_5 = "Compare the canonical power scaling or feats of Goku, Naruto Uzumaki, and Ichigo Kurosaki at their strongest confirmed forms. Visualise destructive capability tiers. Mark the strongest based on feats only. Then apply a canonical power multiplier from their series (e.g., form transformation) and re-rank then visualise."

EXAMPLE_6 = "Compare streaming numbers, Grammy wins, and Billboard Hot 100 entries of Kendrick Lamar, Drake, and J. Cole. Visualise them. Mark the statistically most awarded artist. Then find a real remix collaboration that boosted streaming numbers and model its impact across all three."

EXAMPLE_7 = "Who are the 3 most likely teams to win the UEFA Champions League. Rank them and use a barchart to visualise the odds."

EXAMPLE_8 = """Compare Satoru Gojo, Ryomen Sukuna, and Megumi Fushiguro in terms of:
    1. Domain Expansion mechanics
    2. Energy cost
    3. Domain range
    4. Lethality conditions
Visualise comparative strengths. Then simulate overlapping domain activation and determine the likely outcome under canon rules."""

EXAMPLE_9 = """You are playing Yu-Gi-Oh under Duel Links / Speed Duel rules:

- Starting LP: 4000
- 3 Monster Zones
- 3 Spell/Trap Zones
- Standard TCG rulings apply
- You are only allowed to look up information on cards and their effects. Nothing else.

It is your turn. You have not yet entered the Battle Phase.

CURRENT BOARD STATE

Opponent:
- LP: 4000
- Field: No monsters
- Graveyard contains exactly 3 Dragon monsters:
  - Blue-Eyes White Dragon
  - Red-Eyes Black Dragon
  - Luster Dragon

You:
- LP: 4000
- Field (face-up):
  - Buster Blader

- Hand:
  - Dark Magician
  - Polymerization

- Extra Deck:
  - Dark Paladin

TASK

You can win this turn by playing exactly one card from your hand.

1. Identify which card you must play.
2. Explain why it results in lethal damage this turn.
3. Show all attack calculations explicitly.
4. Use only official card effects (look these up).
5. Do not assume any additional cards or hidden effects.

Your final answer must be structured in a nicely designed graphic with the following text exactly as:

Card Played:
Reasoning:
ATK Calculation:
Final Damage:
"""

EXAMPLE_10 = """Of the authors (First M. Last) that worked on the paper "Pie Menus or Linear Menus, Which Is Better?" in 2015, what was the title of the first paper authored by the one that had authored prior papers?"""
result = dispatch(
    user_input=EXAMPLE_8,
    turn_length=None,
    model="openai/gpt-oss-20b", #"openai/gpt-oss-20b", "Qwen/Qwen3-8B"
    reasoning_effort="high",
    verbose=True
)

print("\n📜 Full Trace Tree:")
result["trace"].pretty_print()

# Save trace to disk
path = result["trace"].save()
print(f"\n💾 Trace saved to: {path}")