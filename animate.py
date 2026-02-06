from utils.plotting import dataset_to_gif

# After generating data, e.g. python generate_data.py case=cylinder steps=500
dataset_to_gif(
    "data_base/cylinder_case.h5",
    "data_base/cylinder_evolution.gif",
    "cylinder",
    step_skip=5,   # every 5th step to keep file smaller
    duration_ms=80,
)