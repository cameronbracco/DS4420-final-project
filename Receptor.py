class Receptor:

    def __init__(self, index):
        self.index = index

        self.activation = 0
    
    def update(self, newSample):
        self.activation = newSample[self.index]
    