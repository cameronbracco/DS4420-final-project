from torch import zeros, full, logical_not, Tensor


class Quantilizer:
    
    # Creates a quantilizer initialized at zero that will move specific amounts
    # when provided a new value 
    def __init__(self, n_outputs, step_up, step_down, device='cuda'):
        self.n_outputs = n_outputs
        self.step_ups = full((n_outputs, ), step_up, device=device)
        self.step_downs = full((n_outputs, ), step_down, device=device)

        self.curr_steps = zeros((n_outputs, ), device=device)
    
    # Checks if a value "passes" the quantilizer and moves the internal step counter accordingly
    # True if strictly greater than the current step, false otherwise
    def check_negs(self, vals: Tensor, true_idx) -> Tensor:
        mask_up = vals > self.curr_steps
        mask_up[true_idx] = False
        self.curr_steps[mask_up] += self.step_ups[mask_up]
        mask_down = logical_not(mask_up)
        mask_down[true_idx] = False
        self.curr_steps[mask_down] -= self.step_downs[mask_down]
        return mask_up

    def check_pos(self, val, true_idx) -> bool:
        if val > self.curr_steps[true_idx]:
            self.curr_steps[true_idx] += self.step_ups[true_idx]
            return True
        else:
            self.curr_steps[true_idx] -= self.step_downs[true_idx]
            return False
