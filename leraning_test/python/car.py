
class Car():

    def __init__(self, make, model, year, remain_gas=0):
        self.year = year
        self.model = model
        self.make = make
        self.remain_gas = remain_gas

    def describe_car(self):
        print(self.year, ' ', self.make, ' Model ', self.model)

    def print_remain_gas(self):
        print("There is ", self.remain_gas, "L gas remained")
