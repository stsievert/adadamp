class SimpleBaseDamper:    
    def __init__(self, initial=128):
        self.initial = initial
        
    def damping(self):
        return self.initial

class SimpleGeoDamp(SimpleBaseDamper):
    def __init__(self, initial=128, dwell=5, factor=5):
        self.initial = initial
        self.dwell = dwell
        self.factor = 5

    def damping(self, meta):
        return self.initial * (self.factor ** meta["epochs"] // self.dwell) # I think