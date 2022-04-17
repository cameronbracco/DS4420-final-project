class Quantilizer:
    
    # Creates a quantilizer initialized at zero that will move specific amounts
    # when provided a new value 
    def __init__(self, stepUp, stepDown):
        self.stepUp = stepUp
        self.stepDown = stepDown

        self.currStep = 0
    
    # Checks if a value "passes" the quantilizer and moves the internal step counter accordingly
    # True if strictly greater than the current step, false otherwise
    def check(self, val):
        if val > self.currStep:
            self.currStep += self.stepUp
            
            return True
        else:
            self.currStep -= self.stepDown
            return False
