import numpy as np

def save_state_dict(state_dict: dict, path: str):
    # state_dict is {name: np.ndarray}
    np.savez(path, **state_dict)

def load_state_dict(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
