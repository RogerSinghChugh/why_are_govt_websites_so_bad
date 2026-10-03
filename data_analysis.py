from pathlib import Path
from clustimage import Clustimage

dataset_path = str(Path("resources/pictures"))

cl = Clustimage()

# You can pass the folder directly; no need to call import_data separately.
results = cl.fit_transform(dataset_path)

# Unique samples (center image per cluster, etc.)
unique_samples = cl.unique()
print(unique_samples.keys())  # dict_keys(['labels', 'idx', 'xycoord_center', 'pathnames', 'img_mean'])

# Collect the unique files (paths) using the indices
unique_idxs = unique_samples['idx']
unique_paths = [results['pathnames'][i] for i in unique_idxs]
print("Unique indices:", unique_idxs)
print("Unique files:", unique_paths)

# Plot
cl.plot_unique()
