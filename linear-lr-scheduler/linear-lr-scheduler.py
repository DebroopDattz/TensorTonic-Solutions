def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:

   
    if warmup_steps > 0 and step < warmup_steps:
        return initial_lr * step / warmup_steps

   
    if step >= total_steps:
        return final_lr


    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    return initial_lr - progress * (initial_lr - final_lr)
