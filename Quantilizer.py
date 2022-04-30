class Quantilizer:
    
    # Creates a quantilizer initialized at zero that will move specific amounts
    # when provided a new value 
    def __init__(self, step_up, step_down, initial_curr_step=0.0):
        self.step_up = step_up
        self.step_down = step_down

        self.curr_step = initial_curr_step
    
    # Checks if a value "passes" the quantilizer and moves the internal step counter accordingly
    # True if strictly greater than the current step, false otherwise
    def check(self, val):
        if val < self.curr_step:
            self.curr_step -= self.step_down
            return False
        elif val > self.curr_step:
            self.curr_step += self.step_up
        return True
