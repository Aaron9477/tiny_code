from car import Car
from battary import Battary

class ElectricCar(Car):

    def __init__(self, make, model, year, battary_size=70):
        super().__init__(make, model, year)
        self.battary = Battary(battary_size)

    def print_remain_gas(self):
        print("This is electric car, no gas!")