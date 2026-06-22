from RainChanceTrain import weight_1
from RainChanceTrain import weight_2
from RainChanceTrain import bias
from math import e

TempRange = int(input("temperature range: "))
AvgTemp = int(input("average temperature: "))

z = bias + (weight_1 * TempRange) + (weight_2 * AvgTemp)
predict = 1 / (1 + e ** (0-z))

print(predict)
