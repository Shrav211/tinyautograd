import numpy as np

def save_state_dict(state_dict: dict, path: str):
    # state_dict is {name: np.ndarray}
    np.savez(path, **state_dict)

def load_state_dict(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}

def save_checkpoint(path, model, opt=None, meta=None):
    """
    Saves a single .npz containing:
      - model_state (dict name->np.ndarray)
      - opt_state   (dict of python objects/arrays)
      - meta        (dict)
    """
    if meta is None:
        meta = {}

    ckpt = {
        "format_version": 1,
        "meta": meta,
        "model_state": model.state_dict(),          # dict[str] -> np.ndarray
        "opt_state": opt.state_dict() if opt is not None else None,
    }

    # Need allow_pickle because ckpt contains nested dicts/lists
    np.savez(path, ckpt=np.array(ckpt, dtype=object))

def load_checkpoint(path, model, opt=None, strict=True):
    """
    Loads checkpoint saved by save_checkpoint.
    Returns the loaded checkpoint dict (meta etc.).
    """
    z = np.load(path, allow_pickle=True)
    ckpt = z["ckpt"].item()

    model_state = ckpt.get("model_state", {})
    model.load_state_dict(model_state, strict=strict)

    opt_state = ckpt.get("opt_state", None)
    if opt is not None and opt_state is not None:
        opt.load_state_dict(opt_state)

    return ckpt