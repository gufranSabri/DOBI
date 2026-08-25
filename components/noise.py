import torch


def response_mask_from_labels(labels):
    """Recover the TRUE response-token positions (in input_ids space) from the data
    loader's PRE-SHIFTED label mask.

    The loader sets labels[t] == input_ids[t+1] on the scored span, so `labels != -100`
    marks indices [len(ctx)-1 .. len(ctx)+len(resp)-1] — the response span shifted LEFT
    by one (it starts at the last context token and ends at the last response token).
    The real response tokens sit at input_ids indices [len(ctx) .. len(ctx)+len(resp)-1],
    i.e. the label mask shifted RIGHT by one. Rolling right by one recovers them.

    Returns (B, T) bool: True exactly on the response tokens in input_ids.
    """
    lab = labels != -100                                  # (B, T) pre-shifted mask
    resp = torch.zeros_like(lab)
    resp[:, 1:] = lab[:, :-1]                             # shift right by one
    return resp
