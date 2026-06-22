from RainChanceTrain import weight_1
from RainChanceTrain import weight_2
from RainChanceTrain import bias
from RainChanceDataPrep import X1
from RainChanceDataPrep import X1
from math import e

TempRange = int(input("temperature range: "))
AvgTemp = int(input("average temperature: "))

TempRange = (TempRange - X1.min())/(X1.max() - X1.min())
TempRange = (AvgTemp - X1.min())/(X1.max() - X1.min())

z = bias + (weight_1 * TempRange) + (weight_2 * AvgTemp)
predict = 1 / (1 + e ** (0-z))

print(predict)
