

class Battary():

    def __init__(self, battary_size=70):
        self.battary_size = battary_size

    def describe_battary(self):
        print("The battary size is {a}.".format(a=str(self.battary_size)))