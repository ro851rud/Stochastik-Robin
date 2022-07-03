# Kernel Density Estimation, KDE
import json
import math
from statistics import variance
import statistics
from icecream import ic
# from scipy.stats.stats import pearsonr
from sklearn import linear_model
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
import statsmodels.stats.weightstats
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as st

# boxplot
import seaborn as seaborn
from scipy.stats import binom
from scipy.stats import geom
from scipy.stats import hypergeom
from scipy.stats import nhypergeom
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import uniform
from scipy.integrate import quad


########README#########
# Das aktuellste File liegt hier:
# https://github.com/ju851zel/msi-stochastik
# Das file das hier hochgeladen wurde ist die aktuellste Version stand 14.07.2021.
# Im Zweifel soll aber immer die auf Github gelten.


def pretty_print(pretty_print_content):
    print(json.dumps(pretty_print_content, indent=2, sort_keys=True))


def nUeberK(n,k):
    x = math.comb(n, k)
    print(x)
    return x

def regression(a, b):
    r = empirischeKorrellationsKoeffizient(a, b)
    sy = empirischeNminus1Standardabweichung(b)
    sx = empirischeNminus1Standardabweichung(a)
    xm = arithmMittel(a)
    ym = arithmMittel(b)
    k = r * (sy / sx)
    d = ym - (k * xm)
    print("f(x) = k * x + d")
    print("k = rx,y * (sy/sx)", k)
    print("d = ym - k * xm = ", d)
    plt.scatter(a, b)
    x = np.linspace(min(a), max(a), 100)
    y = k * x + d
    plt.plot(x, y, '-r', label='y=2x+1')
    plt.ylabel('x first')
    plt.xlabel('y second')
    plt.grid(True)
    plt.show()


######## Absolute Haufigkeit ########
# [3, 4, 5, 1, 5, 2, 1, 3, 1, 3]
def absHauf(array): __absoluteHaufigkeit(array)


def __absoluteHaufigkeit(array):
    res = {}
    absolut = {}
    for item in array:
        if item not in res.keys():
            res[item] = 0
        res[item] = res[item] + 1
    for key, val in res.items():
        absolut[key] = val
    plt.bar(absolut.keys(), absolut.values())
    plt.ylabel('absolut Haufigkeit')
    plt.xlabel('nums')
    plt.grid(True)
    plt.show()
    print("Absolute Haufigkeiten: ")
    pretty_print(absolut)
    return absolut


######## Relative Haufigkeit ########
# [3, 4, 5, 1, 5, 2, 1, 3, 1, 3]
def relHauf(array): __relativeHaufigkeit(array)


def __relativeHaufigkeit(array):
    res = {}
    relative = {}
    for item in array:
        if item not in res.keys():
            res[item] = 0
        res[item] = res[item] + 1
    for key, val in res.items():
        relative[key] = val / len(array)
    plt.bar(relative.keys(), relative.values())
    plt.ylabel('relative Haufigkeit')
    plt.xlabel('nums')
    plt.grid(True)
    plt.show()
    print("Relative Haufigkeiten: ")
    pretty_print(relative)
    return relative

def mittelAusZwei(a1, a2):
    x = np.mean(np.array([a1,a2]))
    print(x)


def mittelwert(array):
    x = np.mean(array)
    print("Mittelwert: ", x)
    return x


def arithmMittel(array):
    x = np.mean(array)
    print("ArithMittel: ", x)
    return x


def median(array):
    x = np.median(array)
    print("Median: ", x)
    return x


def modal(array):
    # haufigste auftretende Wert
    x = scipy.stats.mode(array).mode
    print("Modal (haufigst auftret Wert): ", x)
    return x

def quant(array, quanti):
    x = np.quantile(array, quanti)
    # y = np.percentile(array, quanti*100)

    print(x)
    # print(y)

# [1.3, 5.0, 1.3, 2.7, 4.0, 3.7], percentile=50
def pQuantil(array, qnatiel):
    res = 0
    array = sorted(array)
    index = int(len(array) * (qnatiel / 100))
    if qnatiel % 2 == 0:
        res = 0.5 * (array[index - 1] + array[index])
    else:
        index = math.ceil(index)
        res = array[index]
    x = np.quantile(array, qnatiel)
    d = qnatiel *100
    y = np.percentile(array, d)
    # print(x, len(array))

    print(qnatiel, "-Quartil: ", x)
    print(qnatiel, "-Quartil: ", y)
    return res


def empirischNminus1Varianz(array):
    # n-1
    x = np.var(array, ddof=1)
    print("empirische Varianz: ", x)
    return x


def emp_stdw(array):
    return empirischeNminus1Standardabweichung(array)


def empirischeNminus1Standardabweichung(array):
    # n-1
    x = np.std(array, ddof=1)
    print("empirische Standardabweichung: ", x)
    return x


# echte Werte[1, 1, 1, 1, 2, 2, 2, 3]
def diskreteVarianz1(array):
    x, weights = valuesToValAndArray(array)
    diskreteVarianz(x, weights)

def varianz(array):
    # x = statistics.variance(array)
    ic(
        "Diskrete ZV: ",
        np.var(array, ddof=0),
    )
    # print(x)

# xi= [1, 2, 3], weights=[0.5, 0.125, 0.375]
def diskreteVarianz(array, weights):
    # n
    x = statsmodels.stats.weightstats.DescrStatsW(array, weights=weights, ddof=0).var
    print("Varianz einer Zufallsgrosse: ", x)
    return x


def valuesToValAndArray(array):
    res = {}
    relative = {}
    for item in array:
        if item not in res.keys():
            res[item] = 0
        res[item] = res[item] + 1
    for key, val in res.items():
        relative[key] = val / len(array)
    return list(relative.keys()), list(relative.values())


# echte Werte[1, 1, 1, 1, 2, 2, 2, 3]
def diskreteErwartungswert1(array):
    x, weights = valuesToValAndArray(array)
    diskreteErwartungswert(x, weights)


# xi= [1, 2, 3], weights=[0.5, 0.125, 0.375]
def diskreteErwartungswert(array, weights):
    # n
    x = np.average(array, weights=weights)
    print("Erwartungswert einer Zufallsgrosse: ", x)
    return x


# echte Werte[1, 1, 1, 1, 2, 2, 2, 3]
def diskreteStandardabweichung1(array):
    x, weights = valuesToValAndArray(array)
    diskreteStandardabweichung(x, weights)


# xi= [1, 2, 3], weights=[0.5, 0.125, 0.375]
def diskreteStandardabweichung(array, weights):
    # n
    x = statsmodels.stats.weightstats.DescrStatsW(array, weights=weights, ddof=0).std
    print("normale Standardabweichung: ", x)
    return x


def spannweite(array):
    x = max(array) - min(array)
    print("Spannweite: ", x)
    return x


def interquartilabstandIRQ(array):
    x = scipy.stats.iqr(array)
    print("Interquartilabstand: ", x)
    return x


def empirischeSchiefe(array):  # skewness
    x = scipy.stats.skew(array)
    print("empirische Schiefe: ", x)
    return x


def empirischeWoelbung(array):  # kurtosis
    x = scipy.stats.kurtosis(array)
    print("empirische Woelbung: ", x)
    return x


def empirischeKorrellationsKoeffizient(a, b):
    x = np.corrcoef(a, b)
    print("empirischer korrelationskoeff: ", x)
    # print("empirischer korrelationskoeff: ", np.corrcoef(a, b)[0][1])
    # print("pearson korrelationskoeff (p-wert): ", pearsonr(a, b))
    return np.corrcoef(a, b)[0][1]

def empcovari(a,b):
    n = len(a)
    ax = np.mean(a)
    bx = np.mean(b)
    r = 0
    for i in range(n):
        r += (a[i]-ax)*(b[i]-bx)
    
    p = 1/(n-1)*r
    print(p)
    return p

def empirischeKovarianz(a, b):
    data = np.array([a, b])
    print("empirischer kovarianz ", np.cov(data, bias=True))


######## Empirische Verteilungsfunktion ########
# CFD cumulative distribution function
# [3, 4, 5, 1, 5, 2, 1, 3, 1, 3]
def empirischeVerteilungsfunktion(data):
    relative = __relativeHaufigkeit(data)
    keys = []
    vals = []
    for key, val in relative.items():
        keys.append(key)
        vals.append(val)

    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(p, data_sorted)
    ax1.set_xlabel('$p$')
    ax1.set_ylabel('$x$')

    ax2 = fig.add_subplot(122)
    ax2.plot(data_sorted, p)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$p$')
    plt.grid()
    plt.show()
    print("empirischer Verteilungsfunktion ", np.cumsum(data))

def konfidenzinterfallNormalverteilt(a, percent):
    k = st.norm.interval(alpha=percent, loc=a)
    print(k)

def konfidenzintervall(a, percent):
    k = st.t.interval(alpha=percent, df=len(a)-1, loc=np.mean(a), scale=st.sem(a))
    print("Konfidenzintervall: ", k)

# CFD cumulative distribution function
def kde(a, b):
    pass  # todo


def bin_vert(n, p, array, verteilung=False):
    binomial_verteilung(n, p, array, verteilung)


def binomial_verteilung(n, p, array, verteilung=False):
    mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')
    r_values = list(range(n + 1))
    dist = [binom.pmf(r, n, p) for r in r_values]
    if verteilung:
        for i in range(n + 1):
            print(str(r_values[i]) + "\t" + str(dist[i]))

    sum = 0
    for i in array:
        sum += dist[i]
    print("Wahrsch: ", array, " mal: ", sum)
    print("Erwartungswert", mean)
    print("Varianz", var)
    print("Schiefe", skew)
    print("Wölbung", kurt)

def nhyperGeomVert(total, elements, trys):
    [M, n, N] = [elements, trys, elements]
    p = hypergeom(M, n, N)
    print(p)

   

def hyp_geom_vert(array, nStichprobe, elementeMitEigenschaftM, elementeN):
    hyper_geometrisch_verteilung(array, nStichprobe, elementeMitEigenschaftM, elementeN)
    


def hyper_geometrisch_verteilung(array, nStichprobe, elementeMitEigenschaftM, elementeN):
    mean, var, skew, kurt = hypergeom.stats(elementeN, nStichprobe, elementeMitEigenschaftM, moments='mvsk')
    sum = 0
    for i in array:
        sum += scipy.stats.distributions.hypergeom.pmf(i, elementeN, nStichprobe, elementeMitEigenschaftM)
    print("Wahrscheinlichkeit", sum)
    print("Erwartungswert", mean)
    print("Varianz", var)
    print("Schiefe", skew)
    print("Wölbung", kurt)
    # print("Wahrsch: Aus", nElements, "Elementen, wovon", mEigenschaft, "eine Eigenschaft haben", kZiehen, "zu ziehen: ",


def geom_vert(p, array, quantil=None):
    geometrisch_verteilung(p, array, quantil)


def geometrisch_verteilung(p, array, quantil=None):
    mean, var, skew, kurt = geom.stats(p, moments='mvsk')
    sum = 0
    for i in array:
        sum += scipy.stats.distributions.geom.pmf(i, p)
    print("Wahrscheinlichkeit", sum)
    print("Erwartungswert", mean)
    print("Varianz", var)
    print("Schiefe", skew)
    print("Wölbung", kurt)
    if quantil is not None:
        print("p Quantil", scipy.stats.distributions.geom(p).ppf(quantil))

def poissonpmf(k, mu):
    x = poisson.pmf(k, mu)
    print(x)
    return x

def poissoncdf(k,mu):
    x = poisson.cdf(k,mu)
    print(x)
    return x

def pois_vert(rate, array):
    poisson_verteilung(rate, array)


def poisson_verteilung(rate, array):
    mean, var, skew, kurt = poisson.stats(rate, moments='mvsk')

    sum = 0
    for i in array:
        sum += scipy.stats.distributions.poisson.pmf(i, rate)
    print("Wahrscheinlichkeit", sum)
    print("Erwartungswert", mean)
    print("Varianz", var)
    print("Schiefe", skew)
    print("Wölbung", kurt)


def stet_gleich_vert(min, max, x, quantil=None):
    stetige_gleich_verteilung(min, max, x, quantil)


def stetige_gleich_verteilung(min, max, x, quantil=None):
    Verteilungsdichte = norm(min, max).pdf(x)
    Verteilungsfunktion = norm(min, max).cdf(x)
    erwartungswert = norm(min, max).expect()
    variance = norm(min, max).var()

    print("Verteilungsdichte", Verteilungsdichte)
    print("Wahrscheinlichkeit fur x < ", x, ":", Verteilungsfunktion)
    print("Verteilungsfunktion", Verteilungsfunktion)
    if quantil is not None:
        print("p-Quantil", norm(min, max).ppf(quantil))
    print("Erwartungswert", erwartungswert)
    print("Variance", variance)
    print("Standardabweichung", np.sqrt(variance))


def stet_exp_vert(rate, x, quantil=None):
    stetige_exponential_verteilung(rate, x, quantil)


def stetige_exponential_verteilung(rate, x, quantil=None):
    Verteilungsdichte = expon(scale=rate).pdf(x)
    Verteilungsfunktion = expon(scale=rate).cdf(x)
    erwartungswert = expon(scale=rate).expect()
    variance = expon(scale=rate).var()

    print("Verteilungsdichte", Verteilungsdichte)
    print("Wahrscheinlichkeit fur x < ", x, ":", Verteilungsfunktion)
    print("Verteilungsfunktion", Verteilungsfunktion)
    if quantil is not None:
        print("p-Quantil", expon(scale=rate).ppf(quantil))
    print("Erwartungswert", erwartungswert)
    print("Variance", variance)
    print("Standardabweichung", np.sqrt(variance))


def ste_nor_vert(mu, sigma, x, quantil=None):
    stetige_normal_verteilung(mu, sigma, x, quantil)


def stetige_normal_verteilung(mu, sigma, x, quantil=None):
    Verteilungsdichte = norm(mu, sigma).pdf(x)
    Verteilungsfunktion = norm(mu, sigma).cdf(x)
    erwartungswert = norm(mu, sigma).expect()
    variance = norm(mu, sigma).var()
    print("Verteilungsdichte", Verteilungsdichte)
    print("Wahrscheinlichkeit fur x < ", x, ":", Verteilungsfunktion)
    print("Verteilungsfunktion", Verteilungsfunktion)
    if quantil is not None:
        print("p-Quantil", norm(mu, sigma).ppf(quantil))
    print("Erwartungswert", erwartungswert)
    print("Variance", variance)
    print("Standardabweichung", np.sqrt(variance))

def linearReggression(x, y):
    a = np.array(x).reshape(-1, 1)
    b = y
    lm = linear_model.LinearRegression()
    lm.fit(a, b) # fitting the model
    print("The coefficient is:", lm.coef_)
    print("The intercept is:",lm.intercept_)
    lm.fit(a, b) 
    plt.scatter(a, b, color = "r",marker = "o", s = 30)
    y_pred = lm.predict(a)
    plt.plot(a, y_pred, color = "k")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Simple Linear Regression")
    plt.show()


# TESTS############################
def absoluteHaufigkeitTest():
    assert (__absoluteHaufigkeit([1, 2, 2, 4, 5]) == {1: 1, 2: 2, 4: 1, 5: 1})
    assert (__absoluteHaufigkeit([1, 2, 2, 3, 3, 3]) == {1: 1, 2: 2, 3: 3})
    assert (__absoluteHaufigkeit([2, 1, 5, 3, 1, 5]) == {1: 2, 2: 1, 3: 1, 5: 2})


def relativeHaufigkeitTest():
    assert (__relativeHaufigkeit([1, 2, 2, 4, 5]) == {1: 1 / 5, 2: 2 / 5, 4: 1 / 5, 5: 1 / 5})
    assert (__relativeHaufigkeit([1, 2, 2, 3, 3, 3]) == {1: 1 / 6, 2: 2 / 6, 3: 3 / 6})
    assert (__relativeHaufigkeit([2, 1, 5, 3, 1, 5, 6]) == {1: 2 / 7, 2: 1 / 7, 3: 1 / 7, 5: 2 / 7, 6: 1 / 7})


# def empirirscheVerteilungsfunktionTest():
#     assert (empirirscheVerteilungsfunktion([1, 2, 3, 4, 5]) == {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0})


def mittelwertTest():
    assert (mittelwert([1, 2, 3, 4, 5]) == 3)
    assert (mittelwert([1, 1, 1, 1, 2]) == 1.2)
    assert (mittelwert([5, 0, 295]) == 100)


def arithmMittelTest():
    assert (arithmMittel([1, 2, 3, 4, 5]) == 3)
    assert (arithmMittel([1, 1, 1, 1, 2]) == 1.2)
    assert (arithmMittel([5, 0, 295]) == 100)


def medianTest():
    assert (median([1, 2, 3, 4, 5]) == 3)
    assert (median([1, 1, 1, 1, 2]) == 1)
    assert (median([5, 0, 295]) == 5)


def modalTest():
    assert (modal([1, 2, 3, 3, 3, 4, 5]) == [3])
    assert (modal([1, 1, 1, 1, 2]) == 1)
    # todo more than 2 modalvalue gives first


def pQuantilTest():
    list = [14, 15, 17, 18, 19, 22, 24, 29, 36, 41]
    assert (pQuantil(list, 25) == 17)
    assert (pQuantil(list, 50) == 20.5)
    assert (pQuantil(list, 75) == 29)
    assert (pQuantil(list, 90) == 38.5)
    # todo wrong


def empirischNminus1VarianzTest():
    assert (empirischNminus1Varianz([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 2.8)
    # assert (empirischNminus1Varianz([-2, -1, 0, 1, 2]) == 2.5)
    # assert (empirischNminus1Varianz([-20, -10, 0, 10, 20]) == 250)


def empirischeNminus1StandardabweichungTest():
    assert (empirischeNminus1Standardabweichung([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 1.6733200530681511)
    # assert (empirischeNminus1Standardabweichung([-2, -1, 0, 1, 2]) == 1.5811388300841898)
    # assert (empirischeNminus1Standardabweichung([-20, -10, 0, 10, 20]) == 15.811388300841896)


def normaleVarianzTest():
    pass
    # assert (normaleVarianz([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 2.8)
    # assert (empirischNminus1Varianz([-2, -1, 0, 1, 2]) == 2.5)
    # assert (empirischNminus1Varianz([-20, -10, 0, 10, 20]) == 250)


def normaleStandardabweichungTest():
    pass
    # assert (normaleStandardabweichung([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 1.6733200530681511)
    # assert (empirischeNminus1Standardabweichung([-2, -1, 0, 1, 2]) == 1.5811388300841898)
    # assert (empirischeNminus1Standardabweichung([-20, -10, 0, 10, 20]) == 15.811388300841896)


def spannweiteTest():
    assert (spannweite([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 6)
    assert (spannweite([1, 1, 2, 2, 4, 4, 7, 3, 3, 3, 3]) == 6)


def interquartilabstandIRQTest():
    assert (interquartilabstandIRQ([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]))
    # todo wrong


if __name__ == '__main__':
    x = [1.3, 5.0, 1.3, 2.7, 4.0, 3.7]
    y = [2.0, 3.0, 1.0, 2.0, 1.0, 2.7]
    z = [25, 17, 25, 29, 20, 15, 11, 17, 16, 16]
    w = [x,y]
    a = [1, 1, 1, 1, 2, 3, 3, 3]
    b = [1, 1, 1, 2, 2, 2, 3, 3]
    # modal(z)
    # arithmMittel(z)
    # median(z)
    # quant(z, [0.25, 0.5, 0.75, 0.9])
    # emp_stdw(z)
    # interquartilabstandIRQ(z)
    # spannweite(z)
    # mittelwert(x)
    # mittelAusZwei(x,y)
    varianz(a)
    varianz(b)
    # empirischeKorrellationsKoeffizient(y, x)
    # empcovari(x,y)
    # empirischeKovarianz(x,y)
    # konfidenzintervall(y, 0.9)
    # konfidenzinterfallNormalverteilt(10, 0.95)
    # nhyperGeomVert(14, 6, 2)
    # ic(
    #     1/nUeberK(20, 5)* 10000000
    # )
    # ic(poissonpmf(2, 7))
    # ic(poisson.cdf(8,7) - poisson.cdf(5,7))
    # ic(poisson.pmf(0.9, 7))
    # print(geom(0.55).ppf(0.9))
    # print(np.corrcoef([[1.3, 5.0, 1.3, 2.7, 4.0, 3.7], [2.0, 3.0, 1.0, 2.0, 1.0, 2.7]]))
    # empirischeKorrellationsKoeffizient([10, 50, 30, 70, 80, 60, 90, 40, 10, 20, 30, 50, 60], [4, 5, 2, 6, 6, 8, 7, 2, 7, 3, 5, 1, 3])
    # linearReggression([9, 13, 15, 18, 20], [18, 37, 61, 125, 59])
    