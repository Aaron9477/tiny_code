from car import Car
from electric_car import ElectricCar

if __name__ == '__main__':
    my_car = Car("Benz", "S1", 2016, 105)
    my_new_car = ElectricCar("Tesla", "X3", 2017, battary_size=100)

    my_car.describe_car()
    my_new_car.describe_car()

    my_car.print_remain_gas()
    my_new_car.print_remain_gas()

    my_new_car.battary.describe_battary()


    