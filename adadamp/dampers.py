class BaseDamper:
    def __init__(self, initial=128):
        self.initial = initial

    def damping(self, meta):
        return self.initial


class GeoDamp(BaseDamper):
    def __init__(self, initial=128, dwell=5, factor=5):
        self.initial = initial
        self.dwell = dwell
        self.factor = 5

    def damping(self, meta):
        bs = self.initial * (self.factor ** (meta["_epochs"] // self.dwell))
        return bs
