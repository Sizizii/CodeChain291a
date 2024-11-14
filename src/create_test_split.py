import os
from datasets import load_dataset, load_from_disk

dataset = load_dataset("codeparrot/apps", split="test")
intro_ids = set([4182, 4195, 4281, 4333, 4347, 4426, 4450, 4507, 4514, 4704, 4741,4855, 4873, 4952])
interview_ids = set([2106, 2673, 2923])
competition_ids = set([3070, 3286, 3754])

intro_subset = dataset.filter(lambda example: example['problem_id'] in intro_ids)
interview_subset = dataset.filter(lambda example: example['problem_id'] in interview_ids)
competition_subset = dataset.filter(lambda example: example['problem_id'] in competition_ids)

intro_subset.save_to_disk("../data/test_subset/introductory")
interview_subset.save_to_disk("../data/test_subset/interview")
competition_subset.save_to_disk("../data/test_subset/competition")