from RainChanceTrain import weight_1
from RainChanceTrain import weight_2
from RainChanceTrain import weight_3
from RainChanceTrain import bias
from RainChanceDataPrep import Scale1
from RainChanceDataPrep import Scale2
from RainChanceDataPrep import Scale3
from math import e

TempRange = int(input("temperature range: "))
AvgTemp = int(input("average temperature: "))
Elevation = int(input("elevation: "))

TempRange = (TempRange - Scale1.min())/(Scale1.max() - Scale1.min())
AvgTemp = (AvgTemp - Scale2.min())/(Scale2.max() - Scale2.min())
Elevation = (Elevation - Scale3.min())/(Scale3.max() - Scale3.min())

z = bias + (weight_1 * TempRange) + (weight_2 * AvgTemp) + (weight_3 * Elevation)
predict = 1 / (1 + e ** (0-z))

print(predict)
