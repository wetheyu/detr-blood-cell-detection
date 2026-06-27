from datasets import load_dataset

ds = load_dataset("keremberke/blood-cell-object-detection", name="full")
ds.save_to_disk('data')
