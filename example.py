from trajectorykit import dispatch

result = dispatch(
    user_input="Compare the stats of Blue Eyes White Dragon vs Dark Magician vs TimeWizard Yu-Gi-Oh cards and visualise them.",
    turn_length=5,
    max_tokens=32768,
    verbose=True
)

print("\n📜 Full Trace Tree:")
result["trace"].pretty_print()

# Save trace to disk
path = result["trace"].save()
print(f"\n💾 Trace saved to: {path}")