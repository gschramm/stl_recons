import subprocess
import sys

true_counts_list = [3e6, 1e7, 1e8]
lesion_contrast_list = [2, 4, 10]

for true_counts in true_counts_list:
    for lesion_contrast in lesion_contrast_list:
        cmd = [
            sys.executable,
            "osem.py",
            "--true_counts",
            str(true_counts),
            "--lesion_contrast",
            str(lesion_contrast),
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
