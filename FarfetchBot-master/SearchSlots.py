class SearchSlots():
    def __init__(self):
        self.negative = {}
        self.positive = {}

    def add_positive(self, key, value):
        self.positive[key] = value

        if key in self.negative and self.negative[key] == value:
            del self.negative[key]
    

    def add_negative(self, key, value):
        self.negative[key] = value
        if key in self.positive and self.positive[key] == value:
            del self.positive[key]

    def clean(self):
        self.negative = {}
        self.positive = {}

    def get(self):
        return self.positive, self.negative
