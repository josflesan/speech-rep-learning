class Random:
    """
    A simple linear congruential generator.

    The purpose of having our own generator is to have a central place
    to control randomness.

    """

    def __init__(self, seed=0, m=2147483647, a=1103515245, c=12345):
        self.state = seed
        self.m = m
        self.a = a
        self.c = c

    def rand(self):
        self.state = (self.state * self.a + self.c) % self.m
        return self.state / self.m

    def shuffle(self, ell):
        for i in range(len(ell)):
            j = int(self.rand() * i)
            t = ell[i]
            ell[i] = ell[j]
            ell[j] = t

