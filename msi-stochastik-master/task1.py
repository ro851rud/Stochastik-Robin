import requests
import json
import matplotlib.pyplot as plt
import plotly.express as px
import re
from scipy.ndimage.filters import gaussian_filter1d

import numpy as np
from klausur import *


def pretty_print(pretty_print_content):
    print(json.dumps(pretty_print_content, indent=4, sort_keys=True))


BASE_URL = 'https://swapi.dev/api/'
AUTH_KEY = ''
headers = {'Authorization': 'Bearer ' + AUTH_KEY}


#
def sortfunc(elem):
    return int(elem["passengers"].replace(',', ''))


def basic_data():
    # Water and amount of population korreliert?
    page = 1
    response = requests.get(BASE_URL + "planets").content
    planets = []
    while requests.get(BASE_URL + f"planets/?page={page}").ok:
        content = requests.get(BASE_URL + f"planets/?page={page}").content
        planets += json.loads(content)["results"]
        page += 1

    planet_population = []
    planet_diameter = []
    planet_gravity = []
    planet_names = []
    planet_orbital = []

    planets = list(filter(lambda pl: pl["population"] != "unknown" and
                                     pl["gravity"] != "unknown" and
                                     pl["gravity"] != "standard",
                          planets))
    planets.sort(key=sortfunc)

    for planet in planets:
        planet_names.append(planet["name"])
        planet_diameter.append(planet["diameter"])
        planet_gravity.append(float(re.sub("[^\.1-9]", "", planet["gravity"])))
        # planet_gravity.append(planet["gravity"])
        planet_population.append(planet["population"])
    amount_planets = len(planets)
    # print(f"Anzahl der Planeten : {amount_planets}")
    # print(f"diameter: {planet_diameter}")
    print(f"gravity: {planet_gravity}")

    plt.scatter(planet_gravity, planet_population)
    plt.ylabel('Population')
    plt.xlabel('Gravity')
    plt.grid(True)

    # plt.autoscale()
    # plt.yticks(planet_population,
    #            planet_names)
    plt.show()


def testShips():
    # welche imperialen raumschiffe
    # welche schiffe transportieren am meisten passagiere mit kleiner crew am günstigsten
    page = 1
    ships = []
    while requests.get(BASE_URL + f"starships/?page={page}").ok:
        content = requests.get(BASE_URL + f"starships/?page={page}").content
        ships += json.loads(content)["results"]
        page += 1

    newShips = []
    for ship in ships:
        if ship["passengers"] == "n/a" or ship["passengers"] == "unknown" or ship["cost_in_credits"] == "unknown":
            pass
        else:
            newShips.append(ship)
    ships = newShips
    ships.sort(key=sortfunc)
    passengers = []
    cost_in_credits = []
    for ship in ships:
        # passengers.append(int(re.sub("[^0-9]", "", ship["passengers"])))
        passengers.append(int(ship["passengers"].replace(',', '')))
        cost_in_credits.append(int(ship["cost_in_credits"]))

    passengers.pop()
    cost_in_credits.pop()
    # passengers = list(map(lambda pl: pl["passengers"], ships))
    # cost_in_credits = list(map(lambda pl: pl["cost_in_credits"], ships))

    pretty_print(passengers)
    pretty_print(cost_in_credits)

    plt.scatter(passengers, cost_in_credits)
    plt.ylabel('cost_in_credits')
    plt.xlabel('passengers')
    plt.grid(True)
    plt.ticklabel_format(useOffset=False, style='plain')

    # plt.autoscale()
    # plt.yticks(planet_population,
    #            planet_names)
    plt.show()

    # basic_data()


def testWahlen():
    ships = []
    res = requests.get(f"https://api.dawum.de").content
    res = json.loads(res)
    parliaments = res["Parliaments"]
    institutes = res["Institutes"]
    parties = res["Parties"]
    surveys = res["Surveys"]

    bundestagsumfragen = {}
    for survey in surveys.values():
        if survey["Parliament_ID"] == "0":
            if survey["Institute_ID"] not in bundestagsumfragen:
                bundestagsumfragen[survey["Institute_ID"]] = []
            bundestagsumfragen[survey["Institute_ID"]].append(survey)
    # for kay, value in bundestagsumfragen.items():
    #     pretty_print(f"{kay, len(value) }")

    surveys = bundestagsumfragen["2"]
    # pretty_print(surveys)

    data = {}
    for survey in surveys:
        data[survey["Date"]] = {
            "cdu": survey["Results"]["1"],
            "spd": survey["Results"]["2"],
            "inc": ""
        }

    # pretty_print(dates)
    # pretty_print(cduPoints)
    # plt.scatter(dates, cduPoints)
    # plt.ylabel('cdu werte')
    # plt.xlabel('datum')
    # plt.grid(True)
    # plt.ticklabel_format(useOffset=False, style='plain')
    #
    # plt.autoscale()
    # plt.yticks(planet_population,
    #            planet_names)
    # plt.show()

    res = requests.get(f"https://api.corona-zahlen.org/germany/history/incidence").content
    res = json.loads(res)["data"]
    for dataPoint in res:
        if dataPoint["date"][0:10] in data:
            data[dataPoint["date"][0:10]]["inc"] = dataPoint[
                "weekIncidence"]  # str("%.2f" % dataPoint["deaths"]).replace(".", ",")

    resDeath = []
    resCDU = []
    resSPD = []
    resDate = []
    result = []
    for key, value in data.items():
        if value["inc"] != "":
            resDate.append(key)
            resCDU.append((value["cdu"]))
            resDeath.append((value["inc"]))
            resSPD.append((value["spd"]))
            result.append(value)

    resCDU.reverse()
    resDate.reverse()
    resDeath.reverse()
    resSPD.reverse()
    pretty_print(resDate)

    print(np.corrcoef(resCDU, resDeath, 1))
    # Scatterplot
    plt.scatter(
        x=resDate,
        y=resCDU,
        c="blue")

    plt.title("CDU Umfragewerte")
    plt.xticks(rotation=90)
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60])
    plt.show()

    plt.scatter(
        x=resDate,
        y=resDeath,
        c="red")
    plt.title("Corona Inzidenzen")
    plt.xticks(rotation=90)
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60])

    plt.show()

    plt.scatter(
        x=resDate,
        y=resSPD,
        c="green")
    plt.title("SPD Umfragewerte")
    plt.xticks(rotation=90)
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60])

    plt.show()


def tendierenumfrageinstitute():
    ships = []
    res = requests.get(f"https://api.dawum.de").content
    res = json.loads(res)
    parliaments = res["Parliaments"]
    institutes = res["Institutes"]
    parties = res["Parties"]
    surveys = res["Surveys"]

    bundestagsumfragen = {}
    for survey in surveys.values():
        if survey["Parliament_ID"] == "0":  # bawue
            if survey["Institute_ID"] not in bundestagsumfragen:
                bundestagsumfragen[survey["Institute_ID"]] = []
            bundestagsumfragen[survey["Institute_ID"]].append(survey)
    # for kay, value in bundestagsumfragen.items():
    #     pretty_print(f"{kay, len(value) }")

    # surveys = bundestagsumfragen["2"]

    # pretty_print(bundestagsumfragen)
    data = []
    for value in bundestagsumfragen.values():
        for entry in value:
            if "2020" in entry["Date"]:
                data.append({
                    "date": entry["Date"],
                    "inst": int(entry["Institute_ID"]),
                    "cdu": entry["Results"]["1"],
                    "spd": entry["Results"]["2"],
                    "fdp": entry["Results"]["3"],
                    "grune": entry["Results"]["4"],
                    "linke": entry["Results"]["5"],
                    "afd": entry["Results"]["7"],
                })
            pass
    ergebnis = {}
    for value in data:
        if value["inst"] not in ergebnis:
            ergebnis[value["inst"]] = []
        ergebnis[value["inst"]].append(value)

    x = []
    for key in ergebnis.keys():
        dates = []
        grune = []
        for entry in ergebnis[key]:
            dates.append(entry["date"][0:7])
            grune.append(entry["fdp"])  # here
        x.append((dates, grune))

    pretty_print(x)
    fig, ax = plt.subplots()
    i = 0
    for entry in range(0, len(x), 1):
        if len(x[entry][0]) >= 12:
            ax.scatter(x[entry][0], x[entry][1], label=institutes[str(list(ergebnis.keys())[i])]["Name"], alpha=1)
        i = i + 1

    print(i)
    ax.legend()
    # plt.scatter(datum, grune)
    plt.xticks(rotation=90)
    plt.xticks(range(0, 13, 1))

    # plt.ylabel('cdu werte')
    # plt.xlabel('datum')
    # plt.grid(True)
    # plt.ticklabel_format(useOffset=False, style='plain')
    #
    # plt.autoscale()
    # plt.yticks(planet_population,
    #            planet_names)
    plt.show()


def varianzParteien():
    ships = []
    res = requests.get(f"https://api.dawum.de").content
    res = json.loads(res)
    parliaments = res["Parliaments"]
    institutes = res["Institutes"]
    parties = res["Parties"]
    surveys = res["Surveys"]

    bundestagsumfragen = {}
    for survey in surveys.values():
        if survey["Parliament_ID"] == "0":  # bawue
            if survey["Institute_ID"] not in bundestagsumfragen:
                bundestagsumfragen[survey["Institute_ID"]] = []
            bundestagsumfragen[survey["Institute_ID"]].append(survey)
    # for kay, value in bundestagsumfragen.items():
    #     pretty_print(f"{kay, len(value) }")

    # surveys = bundestagsumfragen["2"]

    # pretty_print(bundestagsumfragen)
    data = []
    for value in bundestagsumfragen.values():
        for entry in value:
            if "2020" in entry["Date"]:
                data.append({
                    "date": entry["Date"],
                    "inst": int(entry["Institute_ID"]),
                    "cdu": entry["Results"]["1"],
                    "spd": entry["Results"]["2"],
                    "fdp": entry["Results"]["3"],
                    "grune": entry["Results"]["4"],
                    "linke": entry["Results"]["5"],
                    "afd": entry["Results"]["7"],
                })
            pass
    ergebnis = {}
    for value in data:
        if value["inst"] not in ergebnis:
            ergebnis[value["inst"]] = []
        ergebnis[value["inst"]].append(value)

    x = []
    for key in ergebnis.keys():
        dates = []
        grune = []
        for entry in ergebnis[key]:
            dates.append(entry["date"][0:7])
            grune.append(entry["linke"])  # here
        x.append({"dates": dates, "grune": grune})

    forsa = x[0]
    data = {}
    for i in range(0, len(forsa["dates"]), 1):
        if forsa["dates"][i] not in data:
            data[forsa["dates"][i]] = []
        data[forsa["dates"][i]].append(forsa["grune"][i])

    for date, values in data.copy().items():
        data[date] = {
            "values": values,
            "avg": mittelwert(values),
            "max": max(values),
            "min": min(values),
            "var": np.var(values) #varianz(values)
        }

    # pretty_print(forsa)

    fig, ax = plt.subplots()

    dates = list(data.keys())
    entries = list(data.items())
    values = list(data.values())
    avgs = list(map(lambda x: x["avg"], values))
    mins = list(map(lambda x: x["min"], values))
    maxs = list(map(lambda x: x["max"], values))
    varsPos = list(map(lambda x: x["var"] + x["avg"], values))
    varsNeg = list(map(lambda x: -x["var"] + x["avg"], values))
    # ax.plot(dates, mins)
    # ax.plot(dates, maxs)


    print(max(avgs)-min(avgs))


    ax.plot(dates, avgs)
    ax.plot(dates, varsPos)
    ax.plot(dates, varsNeg)
    # ax.scatter( forsa["dates"], forsa["grune"])



    plt.xticks(rotation=90)
    plt.xticks(range(0, 13, 1))

    # plt.scatter(datum, grune)
    # plt.ylabel('cdu werte')
    # plt.xlabel('datum')
    # plt.grid(True)
    # plt.ticklabel_format(useOffset=False, style='plain')
    #
    # plt.autoscale()
    # plt.yticks(planet_population,
    #            planet_names)
    plt.show()


testShips()
