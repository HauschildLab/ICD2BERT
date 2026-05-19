future_disease_commands = [
    f"python src/classification.py  --dataset mimic --task future_disease --encoding {encoding} --resample_seed {seed} --verbose True"
    for encoding in range(17)
    for seed in range(10)
] + [
    f"python src/classification.py  --dataset ki_thrust --task future_disease --encoding {encoding} --resample_seed {seed} --verbose True"
    for encoding in range(13)
    for seed in range(10)
]

mort_rehosp_commands = [
    f"python src/classification.py  --dataset mimic --task {task} --encoding {encoding} --resample_seed {seed} --verbose True"
    for task in ["mortality", "rehospitalization"]
    for encoding in range(17)
    for seed in range(10)
] + [
    f"python src/classification.py  --dataset ki_thrust --task {task} --encoding {encoding} --resample_seed {seed} --verbose True"
    for task in ["mortality", "rehospitalization"]
    for encoding in range(13)
    for seed in range(10)
]

print(*future_disease_commands, sep="\n")
print("\n" * 5)
print(*mort_rehosp_commands, sep="\n")

with open("./mort_rehosp_array.sh", "w") as f:
    f.write("\n".join(mort_rehosp_commands) + "\n")

with open("./future_disease_array.sh", "w") as f:
    f.write("\n".join(future_disease_commands) + "\n")
