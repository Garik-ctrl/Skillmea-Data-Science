import datetime
import heapq
from ipaddress import collapse_addresses
from logging import info

import pandas as pd
import os

from pandas import isna
from pandas.core.groupby import groupby
from seaborn._stats import aggregation

print(os.getcwd())

jizdy = pd.read_csv("jizdy.csv")
stanice = pd.read_csv("stanice.csv")
kola = pd.read_csv("kola.csv")
uzivatele = pd.read_csv("uzivatele.csv")
pocasi = pd.read_csv("pocasi.csv")

#PŘÍPRAVA DAT

print(jizdy.info())
print(jizdy.head())

print(jizdy.duplicated().sum())                                 # počet duplicitních jízd 400
print(jizdy["trip_id"].duplicated().sum())                      #unikanti trip_id potvrzeni, že se jedná o 400 duplicit
jizdy_ciste = jizdy.drop_duplicates()
print(jizdy_ciste.info())                                       # počet záznamů 39600

print(jizdy.isna().sum())                                       #chybějící záznamy: user_id  7276
print(
    jizdy[jizdy["user_id"].isna()]["user_type"]                 #turista 7276: všechny chybějící hodnoty vyfiltrovali turistu-nemá účet
    .value_counts())


print(jizdy["start_time"].str.contains(r"\.", na=False).sum())  # počet záznamů datum s tečkami 1415
print(jizdy["end_time"].str.contains(r"\.", na=False).sum())    # 1415

jizdy["start_time"] = pd.to_datetime(
    jizdy["start_time"],
    format="mixed"
)

jizdy["end_time"] = pd.to_datetime(
    jizdy["end_time"],
    format="mixed"
)
print(jizdy.info())

print(jizdy["distance_km"].head())                                  # datový typ změnit na float
jizdy["distance_km"] = (                                            # formát pro desetinné číslo musí být s tečkou
    jizdy["distance_km"]
    .str.replace(",", ".", regex=False)
    .astype(float)
)

jizdy["delka_jizdy"] = jizdy["end_time"] - jizdy["start_time"]
print(jizdy.info())
print(jizdy.head())
print(jizdy.describe())                                               # kontrola dat: záporné hodnoty, min délka jízdy=231 days
print((jizdy["delka_jizdy"] < pd.Timedelta(0)).sum())                 # vyhledání počtu záporných hodnot_143
egativni_jizdy = jizdy[
    jizdy["delka_jizdy"] < pd.Timedelta(0)
]

negativni_jizdy = jizdy[
    jizdy["delka_jizdy"] < pd.Timedelta(0)
]
print(
    negativni_jizdy[
        ["trip_id",
         "start_time",
         "end_time",
         "delka_jizdy"]
    ].head(20)
)
negativni_jizdy["delka_jizdy"].sort_values().head(20)                   # podobný charakter end_time je dříve než start_time
                                                                       # jde o nekonzistentní data, proto budou odstraněna

jizdy = jizdy[
    jizdy["delka_jizdy"] >= pd.Timedelta(0)                             # vynechání řádků se zápornou hodnotou "delka_jizdy"
]
print((jizdy["delka_jizdy"] < pd.Timedelta(0)).sum())


print(stanice.info())                                                   # všechny hodnoty jsou doplněny
print(stanice.head())

print(stanice.duplicated().sum())                                       # žádné duplicity
print(stanice["station_id"].duplicated().sum())                         # ověřena unikátnost "station_id"

print(stanice["opened"].head())
stanice["opened"] = pd.to_datetime(stanice["opened"])                   # změna formátu ["opened"] na typ datum
print(stanice.info())
print(stanice["capacity"].describe())                                   # statistické informace o kapacitě


print(kola.info())
print(kola.head())

print(kola.isna().sum())                                                # žádné chybějící hodnoty
print(kola.duplicated().sum())                                          # žádné duplicitní hodnoty


kola["purchase_date"] = pd.to_datetime(kola["purchase_date"])           # převod na datový typ-datum
kola["last_service"] = pd.to_datetime(kola["last_service"])             # převod na datový typ-datum

print(kola.info())

print(
    (kola["last_service"] < kola["purchase_date"]).sum()                # žádné kolo nebylo servisováno před nákupem
)

print(uzivatele.info())                                                 # birth_year-float
print(uzivatele.head())
print(uzivatele.isna().sum())                                           # chybějící hodnoty_1336-sloupec birth year
print(uzivatele.duplicated().sum())                                     #0

print(uzivatele[uzivatele["birth_year"].isna()]["user_type"].value_counts())  #686NA-"user_typ":pay_as_you_go, 650NA_předplatne
print(uzivatele["birth_year"].describe())                               # min "birth_year" =1900; max"birth_year" =2025
print(uzivatele[uzivatele["birth_year"] > 2016])                        # všichni uživatelé mladší 10 let mají rok narození 2025
print(uzivatele[uzivatele["birth_year"] < 1926])                        # všichni uživatelé starší 100 let mají rok narození 1900

print((uzivatele["birth_year"] == 2025).sum())                          #20 uživatelů s nerálným_příliš nízký rokem narození pro pronájem kola
print((uzivatele["birth_year"] == 1900).sum())                          #40 uživaatelů s nerálným příliš vysokým věkem pro pronájem kola

median_rok = uzivatele["birth_year"].median()                           # median roku narození použit pro vyplnění chybějících hodnot

uzivatele["birth_year"] = (
    uzivatele["birth_year"]
    .fillna(median_rok)
)

print(uzivatele.info())                                                 # kontrola- nulové hodnoty odstraněny

uzivatele = uzivatele[
    (uzivatele["birth_year"] > 1900) &                                  # uživatelé bez odlehlých dat narození
    (uzivatele["birth_year"] < 2025)
]
print(uzivatele.info())

uzivatele["birth_year"] = uzivatele["birth_year"].astype(int)           #změna datového typu u "birth_year" na integr
print(uzivatele.info())

print(pocasi.head())
print(pocasi.info())                                                    # date-změnit z typu object na datetype
print(pocasi.isna().sum())                                              # terp_avg _14 hodnot
print(pocasi.duplicated().sum())                                        # 0

pocasi["date"] = pd.to_datetime(pocasi["date"])

print(pocasi[pocasi["temp_avg"].isna()])                                # zobrazení 14 chybějících hodnot "temp_avg"
pocasi["temp_avg"] = pocasi["temp_avg"].interpolate()                   # doplnění chybějících teplot interpolací
print(pocasi.info())


# SPOJOVÁNÍ TABULEK DO TABULKY JIZDYANL

print(jizdy.columns)
print(uzivatele.columns)
print(kola.columns)
print(stanice.columns)
print(pocasi.columns)

jizdy_an = jizdy.merge(                                                       # tabulka:jizdy_an je propojeni tabulek jizdy a uzivatele
    uzivatele,
    on="user_id",
    how="left"
)
print(jizdy_an.info())

jizdy_anb = jizdy_an.merge(                                                   #tabulka jizdy_anb: je propojení tabulek jizdy, uzivatele, kola
    kola,
    on="bike_id",
    how="left"
)
print(jizdy_anb.info())

jizdy_anb["date"] = jizdy_anb["start_time"].dt.normalize()                     #uprava formatu ["start_time"] pouze na format den
print(jizdy_anb["date"])                                                       #tabulka jizdy_anbp: je propojeni jizdy, uzivatele,pocasi

jizdy_anbp=jizdy_anb.merge(
        pocasi,
        on="date",
        how="left"
)
print(jizdy_anbp.info())

print(
jizdy_anbp[                                                                   #vypsání stanic psaných malým písmem
    jizdy_anbp["start_station_id"]
    != jizdy_anbp["start_station_id"].str.upper()
]["start_station_id"].value_counts()
)

jizdy_anbp["start_station_id"] = (                                            #oprva malých písmen na VELKÁ start_station_id
    jizdy_anbp["start_station_id"]
    .str.upper()
)

jizdy_anbp["end_station_id"] = (                                               #oprva malých písmen na VELKÁ -end_station_id
    jizdy_anbp["end_station_id"]
    .str.upper()
)

analytics = jizdy_anbp.merge(                                                    #analytics: propojeni jizdy, uzivatele, pocasi, stanice
    stanice,
    left_on="start_station_id",
    right_on="station_id",
    how="left"
)
print(analytics.info())


analytics = jizdy_anbp.merge(                                                       #analytics: propojeni jizdy, uzivatele, pocasi, stanice
    stanice,
    left_on="end_station_id",
    right_on="station_id",
    how="left"
)

print(analytics.info())
print(len(analytics))                                               #39857 řádků
print(analytics.isna().sum())                                       #181 NA po připojení stanic- nutno řešit

print(
analytics[
    analytics["station_id"].isna()
]["start_station_id"].unique()
)

print(
set(jizdy_anbp["end_station_id"]) - set(stanice["station_id"]))     #stanice "ST999" není uvedena v referenční tabulce stanic
print((analytics["end_station_id"] == "ST999").sum())               #181 jízd končí v ST999, záznamy ponechámv v evidenci jízd,
                                                                    # ale nebudou zahrnuty do analýzy jednotlivých stanic


#ANALÝZA

print(analytics.info())
print(len(analytics))
print(analytics.head())
print(analytics.columns.tolist())



# 1 SEZÓNNOST A POČASÍ


denni_analyza = (analytics
    .groupby("date")
    .agg(
        pocet_jizd=("trip_id", "count"),
        temp_avg=("temp_avg", "first"),
        rain_mm=("rain_mm", "first")
    )
    .reset_index()
)

print(denni_analyza.head())                                            # NA ve sloupcích "temp_avg" a " rain_mm
print(len(denni_analyza))                                              #594
print(denni_analyza.isna().sum())                                      # 4 NA na záčátku tabulky, smazat


denni_analyza = denni_analyza.dropna(                                  #odstranění NA
    subset=["temp_avg", "rain_mm"]
)
print(denni_analyza.head())
print(len(denni_analyza))

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(                                                               #GRAF SEZÓNNOSTI
    denni_analyza["date"],
    denni_analyza["pocet_jizd"]
)

plt.title("Denní počet jízd")                                           # Silná sezónnost: Zima 10-40 jízd/den vs léto 100-170 jízd/den
plt.xlabel("Datum")                                                     # top sezóna 2025: červen-srpen až cca 200 jízd/den
plt.ylabel("Počet jízd")                                                # top sezóna 2026: červen-srpen až cca 170 jízd/den
plt.grid(True)                                                          # sezóna s nemenším počtem pronájmů prosinec - únor
#
# plt.savefig(                                                            # rok 2026 se jeví jako silnější proti roku 2025
#     "01_denni_jizdy_2025_vs_2026.png",                            # delší období vysoké aktivity
#     dpi=300,
#     bbox_inches="tight"                                                 # vysoká denní kolísavost - předpoklad: vliv dne v týdnu, svátků, počasí
# )
# plt.show()


# Otázka č.1 VLIV TEPOLOTY NA POČET JÍZD

print(                                                                  #corr=0,84 - silná závislost počtu ízd na počasí
    denni_analyza[
        ["pocet_jizd", "temp_avg"]
    ].corr()
)

import matplotlib.pyplot as plt                                         # Vyšší teploty jsou spojeny s vyšším počtem jízd.
plt.figure(figsize=(8,6))                                               # Největší počty jízd se objevují při teplotách 20-25%
                                                                        # Teplota není jediným faktorem, neboť při stejné teplotě jsou různé počty jízd
plt.scatter(
    denni_analyza["temp_avg"],
    denni_analyza["pocet_jizd"]
)

plt.title("Teplota vs počet jízd")
plt.xlabel("Průměrná teplota (°C)")
plt.ylabel("Počet jízd")

plt.grid(True)

# plt.savefig(
#     "02_teplota_vs_pocet_jizd.png",
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()

import numpy as np

denni_analyza["typ_dne"] = np.where(
    denni_analyza["rain_mm"] > 0,
    "destivy",
    "suchy"
)

print(
    denni_analyza
    .groupby("typ_dne")["pocet_jizd"]
    .mean()
)
denni_analyza.groupby(
    "typ_dne"
)["pocet_jizd"].mean().plot(
    kind="bar",
    figsize=(6,4)
)

# plt.title("Průměrný počet jízd podle počasí")                    #deštivý den: průměrně 47,92 jízd
# plt.xlabel("Typ dne")                                            #slunečný den: průměrně 75,36 jízd
# plt.ylabel("Počet jízd")                                         #rozdíl cca 57%
# plt.show()

# Otázka č.2 KDO JEZDÍ A KDY?

analytics["hodina"] = analytics["start_time"].dt.hour             #pomocný sloupec "hodina"

analytics["den_v_tydnu"] = analytics["start_time"].dt.day_name()  #pomocny soupec "den_v_tydnu"

print(analytics.info())
print(
    analytics["user_type_x"].value_counts()                       # uživatelé jsou: "predplatne", "pay_as_you_go", "turista"
)

analytics["skupina"] = np.where(                                  #rezdělení uživatelů do 2 skupin
    analytics["user_type_x"] == "predplatne",
    "clen",
    "ostatni")


clenove = analytics[
    analytics["skupina"] == "clen"
].groupby("hodina").size()

ostatni = analytics[
    analytics["skupina"] == "ostatni"
].groupby("hodina").size()


plt.figure(figsize=(10,5))                                         #vykresleni obou křivek do jednoho grafu

plt.plot(
    clenove.index,                                           #modrá křivka = registrovani uživatelé
    clenove.values,                                                #křivka odpovídá pravidelnému dojíždění
    label="Členové"                                                #ranní špička mezi 7 a 8 hod
)                                                                  #odpolední špička mezi 16 a 17 hod

plt.plot(
    ostatni.index,                                           #oranžová křivka = pay_as_you_go + turista
    ostatni.values,                                                #pozvolný nárůst dopoledne
    label="Ostatní"                                                #neregistrovaní uživatelé jezdí hlavně ve svém volném čase
)

plt.xlabel("Hodina")
plt.ylabel("Počet jízd")
plt.title("Využití sdílených kol během dne")
plt.legend()
plt.grid(True)

# plt.savefig(
#     "03_clenove_vs_ostatni_podle_hodiny.png",
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()

#ZÁVĚR
# ZÁVĚR:
# Počet jízd je silně sezónní a výrazně závisí na počasí.
# Mezi teplotou a počtem jízd byla zjištěna velmi silná
# pozitivní korelace (r = 0,84).
# V suchých dnech proběhlo v průměru 75 jízd denně,
# v deštivých pouze 48 jízd.




#POROVNÁNÍ VYUŽITÍ KOL V PRACOVNÍ DNY A O VÍKENDU
analytics["typ_dne"] = np.where(                                    #rozdělení dnů v týdnu na "víkend" a "pracovní_den"
    analytics["start_time"].dt.weekday >= 5,
    "vikend",
    "pracovni_den"
)

pracovni = (
    analytics[
        analytics["typ_dne"] == "pracovni_den"
    ]
    .groupby("hodina")
    .size()
)

vikend = (
    analytics[
        analytics["typ_dne"] == "vikend"
    ]
    .groupby("hodina")
    .size()
)


plt.figure(figsize=(10,5))                                          #Graf

plt.plot(                                                           #modrá křívka - pracovní den, má dvě výrazné špičky
    pracovni.index,                                           #charakter křivky je stejný u členů
    pracovni.values,
    label="Pracovní den"
)

plt.plot(                                                           #oranžová křivka - víkend
    vikend.index,                                             #ráno minimální počet jízd
    vikend.values,                                                  #pozvolný nárůst během dopolene, vrchol mezi 12:00 -14:00
    label="Víkend"                                                  #odpovídá rekreačnímu využití kol
)

plt.xlabel("Hodina")
plt.ylabel("Počet jízd")
plt.title("Pracovní den vs víkend")
plt.legend()
plt.grid(True)

# plt.savefig(
#     "04_pracovni_den_vs_vikend.png",
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()

# HEATMAPA
analytics["hodina"] = analytics["start_time"].dt.hour               #příprava dat
analytics["den_v_tydnu"] = analytics["start_time"].dt.day_name()

print(analytics["den_v_tydnu"].value_counts())
print(analytics["hodina"].head())

poradi = [                                                          #seřazení dnů v týdnu
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]
analytics["den_v_tydnu"] = pd.Categorical(
    analytics["den_v_tydnu"],
    categories=poradi,
    ordered=True
)

heatmap_data = pd.crosstab(                                         # vytvoření datové matice pro haatmapu
    analytics["den_v_tydnu"],
    analytics["hodina"]
)
print(heatmap_data.head())


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))

sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    annot=False
)

plt.title("Počet jízd podle dne v týdnu a hodiny")                    #pracovní dny mají nejvyšší využití kolem 7:00 a 16:00-17:00
plt.xlabel("Hodina")                                                  #víkendy mají nejvyšší využití kolem 11:00-14:00
plt.ylabel("Den v týdnu")                                             #systém je využíván jinak během pracovního týdne a jinak o víkendech

# plt.savefig(
#     "05_heatmapa_den_hodina.png",
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()


# ZÁVĚR

# ZÁVĚR:
# Členové využívají kola hlavně k dojíždění
# (7–8 h a 16–17 h), zatímco turisté a jednorázoví
# uživatelé jezdí převážně mezi 10.–15. hodinou.
# Ve všední dny dominují dopravní jízdy, o víkendech
# převažuje rekreační využití systému.




#otázka č.3 ZDRAVÍ VOZOVÉHO PARKU


analytics["delka_min"] = (
    analytics["delka_jizdy"]
    .dt.total_seconds() / 60
)

print(analytics.info())
print(analytics["delka_min"].describe())
print(analytics["delka_min"].nlargest(10))                                # extremně dlouhá jízda: 43238.0 min


print(
analytics[
    analytics["delka_min"] < 2                                            #3830 jízd podezřele krátkých, méně než 2min
].shape
)


print(
analytics[
    analytics["delka_min"] < 2
]["bike_id"].value_counts().head(20))                                   #krátké jízdy jsou zaznamanány u různých kol



kratke = analytics[
    analytics["delka_min"] < 2
].groupby("bike_id").size()

vsechny = analytics.groupby("bike_id").size()

print(kratke)                                                           #krátké jízdy (<2min) se vyskytují u 16kol, 3830jízd
print(vsechny)                                                          #všechny jizdy u 400 kol


podil = (
    kratke / vsechny * 100
).sort_values(ascending=False)                                          #podíl krátkých jízd ze všech jízd u 16 podezřelých kol

print(podil.head(20))


podezrela_kola = podil.dropna()                                         # odstranění NA z 16 podezřelých kol

print(podezrela_kola)

srovnani = pd.DataFrame({
    "kratke_jizdy": kratke,
    "vsechny_jizdy": vsechny
})

srovnani["podil_%"] = (                                                 # potvrzení že u 16kol je většina jízd podezřele krátá <2min
    srovnani["kratke_jizdy"] /
    srovnani["vsechny_jizdy"] * 100
)

print(srovnani.sort_values(
    "podil_%",
    ascending=False
))
print(srovnani)


# ODHAD UŠLÝCH PŘÍJMŮ


problemova_kola = podil.dropna().index

print(
analytics[
    analytics["bike_id"].isin(problemova_kola)
]["price_czk"].describe()
)

print(
analytics[
    analytics["bike_id"].isin(problemova_kola)
]["price_czk"].sum()
)


# ZÁVĚR:
# Bylo nalezeno 16 kol, u kterých více než 85 % jízd
# trvalo méně než 2 minuty. Tato kola vykazují výrazně
# odlišné chování od zbytku vozového parku a mohou být
# technicky problematická. Doporučuji jejich servisní kontrolu.



#Otázka č. 4 ZMĚNY CENY 1.6.2026
print(
    analytics["date"].min()                                             #2025-01-03
)
print(
    analytics["date"].max()                                             #2026-10-31
)


analytics["skupina_cena"] = np.where(                                   #vytvoření skupin: "pay_as_you_go" a "ostatních"
    analytics["user_type_x"] == "pay_as_you_go",
    "pay_as_you_go",
    "ostatni"
)

print(
    analytics["skupina_cena"].value_counts()
)

analyza_cena = analytics[
    (
        (analytics["date"] >= "2025-05-01")
        & (analytics["date"] <= "2025-07-31")
    )
    |
    (
        (analytics["date"] >= "2026-05-01")
        & (analytics["date"] <= "2026-07-31")
    )
].copy()
print(analyza_cena.head())
print(analyza_cena.info())

analyza_cena["rok"] = (
    analyza_cena["date"].dt.year
)

print(                                                          #2025 pay_as_you_ go: 2680 jízd, ostatní: 5931 jízd
    analyza_cena                                                #2025 pay_as_you_go: 2572 jízd, ostatní:6983 jízd
    .groupby(
        ["rok", "skupina_cena"]
    )["trip_id"]
    .count()
)
srovnani = analyza_cena.groupby(
    ["rok", "skupina_cena"]
).size().unstack()

srovnani.plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Počet jízd podle skupiny uživatelů")
plt.ylabel("Počet jízd")

plt.savefig(
    "07_dopad_zmeny_ceny.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


paygo_2025 = 2680 / (2680 + 5931) * 100
paygo_2026 = 2572 / (2572 + 6983) * 100

print("2025:", round(paygo_2025, 1), "%")           # 2025: podíl uživatelů pay_as_you_go v roce 31,1
print("2026:", round(paygo_2026, 1), "%")           # 2026: podíl uživatalů pay_as_you_go na všech jízdách kleslna 26,9



#ZÁVĚR:
# Po zdražení od 1. 6. 2026 klesl počet jízd pay_as_you_go
# o 4 %, zatímco ostatní skupiny vzrostly o 18 %.
# Podíl pay_as_you_go na všech jízdách klesl z 31,1 %
# na 26,9 %, což naznačuje možný negativní dopad změny ceny.



