from RainChanceTrain import weight_1
from RainChanceTrain import weight_2
from RainChanceTrain import weight_3
from RainChanceTrain import weight_4
from RainChanceTrain import weight_5
from RainChanceTrain import bias
from RainChanceDataPrep import Scale1
from RainChanceDataPrep import Scale2
from RainChanceDataPrep import Scale3
from RainChanceDataPrep import Scale4
from RainChanceDataPrep import Scale5
from math import e

TempRange = int(input("temperature range: "))
AvgTemp = int(input("average temperature: "))
Elevation = int(input("elevation: "))
Latitude = int(input("latitude: "))
Longitude = int(input("longitude: "))

TempRange = (TempRange - Scale1.min())/(Scale1.max() - Scale1.min())
AvgTemp = (AvgTemp - Scale2.min())/(Scale2.max() - Scale2.min())
Elevation = (Elevation - Scale3.min())/(Scale3.max() - Scale3.min())
Latitude = (Latitude - Scale4.min())/(Scale4.max() - Scale4.min())
Longitude = (Longitude - Scale5.min())/(Scale5.max() - Scale5.min())

z = bias + (weight_1 * TempRange) + (weight_2 * AvgTemp) + (weight_4 * Latitude) + (weight_5 * Longitude)
predict = 1 / (1 + e ** (0-z))

print(predict)
