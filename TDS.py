import math
from scipy.stats import norm, weibull_min, expon
import numpy as np
import statsmodels.api as sm
# from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import statistics
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde

#function that calculated the kde
def calc_kde(feature):
    kde = sm.nonparametric.KDEUnivariate(feature)
    kde.fit()
    return kde

#function that finds the minimum and maximum points, using the kde
def min_max(data):
    kde = calc_kde(data)
    samples = np.linspace(min(data), max(data), 500)
    vals = np.exp(kde.evaluate(samples))
    min_list, max_list = [], []
    theLastMaxIndex = -1
    theLastMinIndex = -1
    for i in range(3, len(vals) - 3, 1):
        if vals[i] == np.max(vals[i - 3: i + 3]): # we check the minimal point in every [i - 3: i + 3]
            max_list.append((samples[i], vals[i]))
            theLastMaxIndex = i
        elif vals[i] == np.min(vals[i - 3: i + 3]): # we check the maximal point in every [i - 3: i + 3]
            min_list.append((samples[i], vals[i]))
            theLastMinIndex = i

        if (len(min_list) == 0) and (len(max_list) != 0): # we take into consideration the first sample
            min_list.append((samples[0], vals[0]))

        if (len(min_list) != 0) and (len(max_list) == 0):
            max_list.append((samples[0], vals[0]))

    if theLastMaxIndex > theLastMinIndex: # we take into consideration the last sample
        min_list.append((samples[len(samples)-1], vals[len(samples)-1]))
    else:
        max_list.append((samples[len(samples)-1], vals[len(samples)-1]))
    return min_list, max_list

#function that present the kde graph
def show_kde(feature):
    kde = calc_kde(feature)
    samples = np.linspace(min(feature), max(feature), 500)
    log_prob = kde.evaluate(samples)
    plt.plot(samples, np.exp(log_prob), c='green')
    plt.title(feature)
    plt.show()

#filtering dataset from not numerical values and more
def filter_dataset(dataset):
    num_features = dataset.select_dtypes(include=[np.number])
    column_names = list(num_features.columns.values)
    uni_features = [f for f in column_names if dataset[f].nunique() > 50 and dataset[f].isna().sum() / len(dataset[f])\
                    < 0.4 and dataset[f].nunique() < len(dataset[f])]
    min_max_features = {}
    for f in uni_features:
        feature = fill_null(dataset[f])
        min_max_features[f] = min_max(feature.to_numpy())
    rel_features = [f for f in min_max_features if len(min_max_features[f][0]) > 1 and len(min_max_features[f][1]) > 1]
    return rel_features

# ks_test for exponential data
def ks_test_exp(empirical):
    params = stats.expon.fit(empirical)
    result = stats.kstest(empirical, lambda x: stats.expon.cdf(x, loc=params[0], scale=params[1]))
    # print(result)
    return result

# ks_test for weibull data
def ks_test_weibull(empirical):
    params = stats.weibull_min.fit(empirical)
    return stats.kstest(empirical, lambda x: stats.weibull_min.cdf(x, c=params[0], loc=params[1], scale=params[2]))

# ks_test for normal data
def ks_test_normal(empirical):
    params = stats.norm.fit(empirical)
    return stats.kstest(empirical, lambda x: stats.norm.cdf(x, loc=params[0], scale=params[1]))

# func mentioned in the paper 
# "A Modified Kolmogorov-Smirnov Test for Normality" 
# in the bottom of page 3
def ksFunc(avg, std, sample):
    n = len(sample)
    finalMaxNumber = float('-inf')
    for i in range(n):
        tempCdf = stats.norm.cdf(((sample[i]-avg) / std), loc=0, scale=1)
        firstNum = (i+1)/n - tempCdf
        secondNum = tempCdf - (i/n)
        tempMax = max(firstNum, secondNum)
        if tempMax > finalMaxNumber:
            finalMaxNumber = tempMax
    return finalMaxNumber

# implementation of the algorithm in the paper 
# "A Modified Kolmogorov-Smirnov Test for Normality" 
# as described in section 3
def ks_test_normal_paper(empirical):
    arr1 = bestL(1 / (2 * len(empirical)), ksFunc(np.mean(empirical), statistics.stdev(empirical), empirical), empirical)
    mean = arr1[0]
    std = arr1[1]
    return stats.kstest(empirical, lambda x: stats.norm.cdf(x, loc=float(mean), scale=float(std)))

# Binary search on the best mean & std parameters,
# that create the minimal L value and it's parameters
def bestL(start, end, feature):
    mean = 2
    std1 = 2
    gl = g((start + end) / 2, feature)  # G(L)
    if (end - start) < 0.001:
        mean = end * np.std(feature) + np.mean(feature)
        stdNew = (gl[1] * np.std(feature)) / math.sqrt(len(feature))
        print("m: " + str(mean) + " s: " + str(stdNew))   # + " t: " + type())
        return [str(mean), str(stdNew)]
    if gl[0] >= 0:
        return bestL(start, (start + end) / 2, feature)
    elif gl[0] < 0:
        return bestL((start + end) / 2, end, feature)

    return [mean, std1]

# the G(L) func in the algorithm in the paper
# "A Modified Kolmogorov-Smirnov Test for Normality" 
def argmax_z(X, L, n):
    def objective(z):
        min_val = np.min([X[kk-1] - z*norm.ppf(kk/n - L) for kk in range(int(np.floor(n*L) + 1), n+1)])
        max_val = np.max([X[kk-1] - z*norm.ppf(L + (kk-1)/n) for kk in range(1, int(np.ceil((1-L)*n)+1))])
        return min_val - max_val
    res = minimize_scalar(lambda z: -objective(z), bounds=(0, max(X)))
    print(objective(res.x))
    print((res.x))
    return objective(res.x), res.x


def g(lvar1, feature):
    return argmax_z(feature, lvar1, len(feature))


# load_dataset
def load_dataset(path, id_col):
    return pd.read_csv(path, index_col=id_col)

# fill null values in feature
def fill_null(feature):
    return feature.fillna(feature.mean())

# generating data for testing
def generate_data(seed=5):
    # rand = np.random.RandomState(seed)
    x = []
    d1 = norm.rvs(loc=903, scale=34, size=2000, random_state=seed)
    x = np.concatenate((x, d1))
    d2 = weibull_min.rvs(5, loc=2034, scale=23, size=2000, random_state=seed)
    x = np.concatenate((x, d2))
    d3 = expon.rvs(loc=4501, scale=8, size=2000, random_state=seed)
    x = np.concatenate((x, d3))
    return x


# return a list thats it's varibles bitween 2 values
def subSetOfDomain(ourFeature, intervalSmallerParm, intervalBigParm):
    # sub_list = datafram[ourFeature].tolist()
    # sub_list.sort()
    sub_list = ourFeature
    listToReturn = list()
    for val in sub_list:
        if (val <= intervalBigParm and val >= intervalSmallerParm):
            listToReturn.append(val)
    return listToReturn

# calculating the weightedArithmeticMean of pvalue for our improved kstest method
def weightedArithmeticMeanPvalues(listOfPvalues, listOfTheirWeights):
    sum = 0
    for i in range(len(listOfPvalues)):
        sum += listOfPvalues[i] * listOfTheirWeights[i]
    return sum


# calculates the pvalue of a given chunk of feature
def pValueEvaluator(feature, begin, end):
    ourInterval = subSetOfDomain(feature, begin, end)
    # print(str(ourInterval))
    if len(ourInterval) / len(feature) > 0.05:
        # print("yes")
        result1 = ks_test_exp(ourInterval).pvalue
        result2 = ks_test_weibull(ourInterval).pvalue
        result3 = ks_test_normal(ourInterval).pvalue
        tempMax = max(result1, result2, result3)
        if tempMax < 0.001:
            return 0, "no theoretical distribution"
        if result1 == tempMax:
            return result1, "Exponential distribution"
        if result2 == tempMax:
            return result2, "Weibull distribution"
        if result3 == tempMax:
            return result3, "Normal distribution" 
    return 0, " "

# changing the subset parameters so it will have a better pvalue and take more space 
def isBetterRangOfPvalue(ourFeature, begin, end):
    baseFeature = (begin, end)
    num1 = updatenumberToCeil(end,  1.1)
    if num1 >= len(ourFeature):
        num1 = len(ourFeature)-1
    num2 = updatenumberToFloor(begin, 0.9)
    if num2 < 0:
        num2 = 0
    num3 = updatenumberToFloor(end, 0.9)
    num4 = updatenumberToCeil(begin, 1.1)

    valueBaseFeature = pValueEvaluator(ourFeature, begin, end)
    valf1 = pValueEvaluator(ourFeature, begin, num1)
    valf2 = pValueEvaluator(ourFeature, num2, end)
    valf3 = pValueEvaluator(ourFeature, begin, num3)
    valf4 = pValueEvaluator(ourFeature, num4, end)

    maxValue = max(valueBaseFeature[0], valf1[0], valf2[0], valf3[0], valf4[0])
    if maxValue == valueBaseFeature[0]:
        return baseFeature, valueBaseFeature
    if maxValue == valf1[0]:
        return (begin, num1), valf1
    if maxValue == valf2[0]:
        return (num2, end), valf2
    if maxValue == valf3[0]:
        return (begin, num3), valf3
    return (num4, end), valf4


def updatenumberToCeil(num, precentageofchange):
    num = num * precentageofchange
    return math.ceil(num)


def updatenumberToFloor(num, precentageofchange):
    num = num * precentageofchange
    return math.floor(num)

# find the best ranges by taking interval bitween every minimal point to the closest minimal point
# maximal point to minimal and minimal to maximal, and make changes on these intervals with changeIntervals func
def bestPvalueRanges(feature):
    min_list, max_list = min_max(feature)
    listOfRanges = list()
    j = 0
    i = 0
    while i < len(min_list) and j < len(max_list):
        if min_list[i][0] < max_list[j][0]:
            listOfRanges.append(((min_list[i][0], max_list[j][0]), pValueEvaluator(feature, min_list[i][0], max_list[j][0])))
            i = i + 1
            if i < len(min_list):
                listOfRanges.append(((min_list[i-1][0], min_list[i][0]), pValueEvaluator(feature, min_list[i-1][0], min_list[i][0])))
        else:
            listOfRanges.append(((max_list[j][0], min_list[i][0]), pValueEvaluator(feature, max_list[j][0], min_list[i][0])))
            j = j + 1
    return changeIntervals(listOfRanges, feature)
    # return recursiveBestPvalueRanges(feature)

#display the distributions graph
def showGraphOfDistrebutions(intervals, feature, name):
    kde = calc_kde(feature)
    samples = np.linspace(0, max(feature), 6000)
    log_prob = kde.evaluate(samples)
    x = np.sort(feature)
    y = np.exp(log_prob)

    colors = ['blue', 'green', 'orange', 'pink', 'red', 'purple', 'pink', 'turquoise']
    figure, axes = plt.subplots()
    for i, interval in enumerate(intervals):
        j = i
        if i > len(colors) - 1:
            j = i % len(colors)
        mask = np.logical_and(samples >= interval[0][0], samples <= interval[0][1])
        plt.plot(samples[mask], np.exp(log_prob[mask]), c=colors[i])
    
    axes.legend()
    # title of the plot
    axes.set_title(str(name))

    # adding distribution names
    for i, interval in enumerate(intervals):
        x_center = np.mean(interval[0])
        y_center = np.max(y) * 1.1
        axes.text(x_center, y_center, f'{interval[1][1]}', ha='center')

    # show the graph
    plt.show()

# looking at the interval starting in the same minimal
# /ending in the same minimal point that overlap and keeping the ones with the best pvalues
def bestIntervalsToCheck(listOfRanges, feature):
    listToReturn = list()
    lengthOfFeature = max(feature) - min(feature)
    isFirstTime = 1
    for i in range(len(listOfRanges)):
        if i+1 < len(listOfRanges) and i+2 < len(listOfRanges):
            if listOfRanges[i][0][0] == listOfRanges[i+1][0][0] and listOfRanges[i+2][0][1] == listOfRanges[i+1][0][1]:
                lengthOfInterval1 = (listOfRanges[i][0][1] - listOfRanges[i][0][0])/lengthOfFeature
                lengthOfInterval2 = (listOfRanges[i+1][0][1] - listOfRanges[i+1][0][0])/lengthOfFeature
                lengthOfInterval3 = (listOfRanges[i+2][0][1] - listOfRanges[i+2][0][0])/lengthOfFeature
                weightedPvalue1 = lengthOfInterval1 * listOfRanges[i][1][0]
                weightedPvalue2 = lengthOfInterval2 * listOfRanges[i+1][1][0]
                weightedPvalue3 = lengthOfInterval3 * listOfRanges[i+2][1][0]
                tempMax = max(weightedPvalue1, weightedPvalue2, weightedPvalue3)
                if weightedPvalue1 == tempMax:
                    listToReturn.append(listOfRanges[i])
                elif weightedPvalue2 == tempMax:
                    listToReturn.append(listOfRanges[i+1])
                elif weightedPvalue3 == tempMax:
                    listToReturn.append(listOfRanges[i+2])
                isFirstTime = 0
                if i+2 == len(listOfRanges) - 1:
                    i = len(listOfRanges)
            elif isFirstTime or i == len(listOfRanges) - 1:
                isFirstTime = 0
                listToReturn.append(listOfRanges[i])
    return listToReturn

#multiplying the edges of interval by 1.1 or 0.9 till it doesnt improve the pvalue
def bestVersionOfInterval(rang, feature):
    pValueinteval1 = rang
    tempPvalue = isBetterRangOfPvalue(feature, rang[0][0], rang[0][1])
    while (tempPvalue[1][0] > pValueinteval1[1][0]):
        pValueinteval1 = tempPvalue
        tempPvalue = isBetterRangOfPvalue(feature, pValueinteval1[0][0], pValueinteval1[0][1])
    return pValueinteval1

# call to bestVersionOfInterval for every interval in the feature
def adjustIntervals(listOfRanges, feature):
    listToReturn = list()
    for rang in listOfRanges:
        rang11 = bestVersionOfInterval(rang, feature)
        if rang11[1][0] != 0:
            listToReturn.append(rang11)
    return listToReturn

# taking care of overlaps
def recursiveCareOfOvelap(feature, rang1, rang2, pvalue1, reqDepth):
    num1 = ((rang1[0][1] - rang2[0][0])/2)
    pValueInterval1 = (rang2[0][0] + num1 - rang1[0][0]) * (pValueEvaluator(feature, rang1[0][0], rang2[0][0] + num1)[0])
    pValueInterval2 = (rang2[0][1] - rang2[0][0] - num1) * (pValueEvaluator(feature, rang2[0][0] + num1, rang2[0][1])[0])
    pvalueNew = pValueInterval1 + pValueInterval2
    if pvalue1 - pvalueNew >= 0.01 or reqDepth >= 300:
        return rang1, rang2
    else:
        rang1 = ((rang1[0][0], rang2[0][0] + num1), pValueEvaluator(feature, rang1[0][0], rang2[0][0] + num1))
        rang2 = ((rang2[0][0] + num1, rang2[0][1]), pValueEvaluator(feature, rang2[0][0] + num1, rang2[0][1]))
        pvalue1 = pvalueNew
        return recursiveCareOfOvelap(feature, rang1, rang2, pvalue1, reqDepth + 1)

# taking care of overlaps
def careOfOverlap(interval1, interval2, feature):
    num1 = ((interval1[0][1] - interval2[0][0])/2)
    pValueInterval1 = (interval2[0][0] + num1 - interval1[0][0]) * (pValueEvaluator(feature, interval1[0][0], interval2[0][0] + num1)[0])
    pValueInterval2 = (interval2[0][1] - interval2[0][0] - num1) * (pValueEvaluator(feature, interval2[0][0] + num1, interval2[0][1])[0])
    pvalueNew = pValueInterval1 + pValueInterval2
    pvalueOriginal = ((interval1[0][1] - interval1[0][0]) * interval1[1][0]) + ((interval2[0][1] - interval1[0][1]) * (pValueEvaluator(feature, interval1[0][1], interval2[0][1])[0]))
    if pvalueNew <= pvalueOriginal:
        rang1 = interval1
        rang2 = ((interval1[0][1], interval2[0][1]), pValueEvaluator(feature, interval1[0][1], interval2[0][1]))
        return rang1, rang2
    else:
        rang1 = ((interval1[0][0], interval2[0][0] + num1), pValueEvaluator(feature, interval1[0][0], interval2[0][0] + num1))
        rang2 = ((interval2[0][0] + num1, interval2[0][1]), pValueEvaluator(feature, interval2[0][0] + num1, interval2[0][1]))
        return recursiveCareOfOvelap(feature, rang1, rang2, pvalueNew, 0)

# taking care of overlaps
def eraseOverlaps(listOfRanges, feature):
    i = 1
    newList = list()
    currentRange = listOfRanges[0]
    while i < len(listOfRanges):
        nextRange = listOfRanges[i]
        if ((currentRange[0][0] <= nextRange[0][0]) and (currentRange[0][1] >= nextRange[0][0])) or ((currentRange[0][0] >= nextRange[0][1]) and (currentRange[0][1] >= nextRange[0][1])):
            rang1, rang2 = careOfOverlap(currentRange, nextRange, feature)
            newList.append(rang1)
            currentRange = rang2
        else:
            newList.append(currentRange)
            currentRange = nextRange
        i = i + 1
    if currentRange not in newList:
        newList.append(currentRange)
    return newList

# taking care of Gaps
def recursiveCareOfGaps(feature, interval1, interval2, pvalue1, reqDepth):
    num1 = ((interval1[0][1] - interval2[0][0])/2)
    pValueInterval1 = (interval1[0][1] + num1 - interval1[0][0]) * (pValueEvaluator(feature, interval1[0][0], interval1[0][1] + num1)[0])
    pValueInterval2 = (interval2[0][1] - interval2[0][0] + num1) * (pValueEvaluator(feature, interval2[0][0] - num1, interval2[0][1])[0])
    pvalueNew = pValueInterval1 + pValueInterval2
    if pvalue1 - pvalueNew >= 0.000001 or reqDepth>=300:
        return interval1, interval2
    else:
        rang1 = ((interval1[0][0], interval1[0][1] + num1), pValueEvaluator(feature, interval1[0][0], interval1[0][1] + num1))
        rang2 = ((interval2[0][0] - num1, interval2[0][1]), pValueEvaluator(feature, interval2[0][0] - num1, interval2[0][1]))
        pvalue1 = pvalueNew
        return recursiveCareOfGaps(feature, rang1, rang2, pvalueNew, reqDepth + 1)


# taking care of Gaps
def careOfGaps(interval1, interval2, feature):
    num1 = ((interval2[0][0] - interval1[0][1])/2)
    pValueInterval1 = (interval1[0][1] + num1 - interval1[0][0]) * (pValueEvaluator(feature, interval1[0][0], interval1[0][1] + num1)[0])
    pValueInterval2 = (interval2[0][1] - interval2[0][0] + num1) * (pValueEvaluator(feature, interval2[0][0] - num1, interval2[0][1])[0])
    pvalueNew = pValueInterval1 + pValueInterval2
    pvalueOriginal1 = ((interval1[0][1] - interval1[0][0]) * interval1[1][0]) + ((interval2[0][1] - interval1[0][1]) * (pValueEvaluator(feature, interval1[0][1], interval2[0][1])[0]))
    pvalueOriginal2 = ((interval2[0][1] - interval2[0][0]) * interval2[1][0]) + ((interval2[0][0] - interval1[0][0]) * (pValueEvaluator(feature, interval1[0][0], interval2[0][0])[0]))
    tempMax = max(pvalueOriginal1, pvalueOriginal2, pvalueNew)
    if pvalueOriginal1 == tempMax:
        rang1 = interval1
        rang2 = ((interval1[0][1], interval2[0][1]), pValueEvaluator(feature, interval1[0][1], interval2[0][1]))
        return rang1, rang2
    elif pvalueOriginal2 == tempMax:
        rang2 = interval2
        rang1 = (interval1[0][0], interval2[0][0]), pValueEvaluator(feature, interval1[0][0], interval2[0][0])
        return rang1, rang2
    else:
        rang1 = ((interval1[0][0], interval1[0][1] + num1), pValueEvaluator(feature, interval1[0][0], interval1[0][1] + num1))
        rang2 = ((interval2[0][0] - num1, interval2[0][1]), pValueEvaluator(feature, interval2[0][0] - num1, interval2[0][1]))
        return recursiveCareOfGaps(feature, rang1, rang2, pvalueNew, 0)



# taking care of Gaps
def eraseGaps(listOfRanges, feature):  # need to fix, not ready
    i = 1
    newList = list()
    currentRange = listOfRanges[0]
    if currentRange[0][0] != 0:
        currentRange = ((0, currentRange[0][1]), pValueEvaluator(feature, 0, currentRange[0][1]))
    while i < len(listOfRanges):
        nextRange = listOfRanges[i]
        if currentRange[0][1] < nextRange[0][1]:
            rang1, rang2 = careOfGaps(currentRange, nextRange, feature)
            newList.append(rang1)
            currentRange = rang2
        else:
            newList.append(currentRange)
            currentRange = nextRange
        i = i + 1
    if currentRange not in newList:
        newList.append(currentRange)
    return newList

#make all the changes on interval so that no gaps, overlaps and the best pvalue for every interval
def changeIntervals(listOfRanges, feature):
    listOfRangesUpdated = bestIntervalsToCheck(listOfRanges, feature)
    listOfRangesUpdated = adjustIntervals(listOfRangesUpdated, feature)
    listOfRangesUpdated = eraseOverlaps(listOfRangesUpdated, feature)
    listOfRangesUpdated = eraseGaps(listOfRangesUpdated, feature)
    return listOfRangesUpdated

# calculate our pvalue for our upgraded kstest
def upgradedKsCalc(feature):
    ranges = bestPvalueRanges(feature)
    listOfWeights = list()
    listOfPvalues = list()
    lengthOfFeature = max(feature)
    for i in range(len(ranges)):
        listOfPvalues.append(ranges[i][1][0])
        listOfWeights.append((ranges[i][0][1] - ranges[i][0][0])/lengthOfFeature)
    pvalueOfFeature = weightedArithmeticMeanPvalues(listOfPvalues, listOfWeights)
    return pvalueOfFeature

# send the result of every solutuon
def run_solutions(feature):
    ourSolution = upgradedKsCalc(feature)
    traditionalSolution = max(ks_test_exp(feature).pvalue, ks_test_normal(feature).pvalue, ks_test_weibull(feature).pvalue)   # stats.kstest(feature)
    return ourSolution, traditionalSolution

# testing the data
def test():
    feature = generate_data()
    ourSolution = upgradedKsCalc(feature)
    traditionalSolution = max(ks_test_exp(feature).pvalue, ks_test_normal(feature).pvalue, ks_test_weibull(feature).pvalue)
    print("our solution: " + str(ourSolution))
    print("original solution: " + str(traditionalSolution))
    ranges = bestPvalueRanges(feature)
    showGraphOfDistrebutions(ranges, feature, "sample")


# testing the graph
def testGraph(feature, name):
    ranges = bestPvalueRanges(feature)
    showGraphOfDistrebutions(ranges, feature, name)


dataset0 = pd.read_csv("./dataset1.csv")
feature0 = dataset0['feature1'].tolist()
# testGraph(feature0, "dataset1")
answer0 = run_solutions(feature0)
print("Dataset 1:")
print("our solution:", str(answer0[0]))
print("taraditional solution:", str(answer0[1]))

dataset1 = pd.read_csv("./dataset2.csv")
feature1 = dataset1['feature1'].tolist()
# testGraph(feature1, "dataset2")
answer1 = run_solutions(feature1)
print("Dataset 2:")
print("our solution:", str(answer1[0]))
print("taraditional solution:", str(answer1[1]))

dataset2 = pd.read_csv("./dataset3.csv")
feature2 = dataset2['feature1'].tolist()
# testGraph(feature2, "dataset3")
answer2 = run_solutions(feature2)
print("Dataset 3:")
print("our solution:", str(answer2[0]))
print("taraditional solution:", str(answer2[1]))

dataset3 = pd.read_csv("./dataset4.csv")
feature3 = dataset3['feature1'].tolist()
# testGraph(feature3, "dataset4")
answer3 = run_solutions(feature3)
print("Dataset 4:")
print("our solution:", str(answer3[0]))
print("taraditional solution:", str(answer3[1]))
