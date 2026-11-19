# =====================================================================================
# VIZUALIZACE DAT III – OKOMENTOVANÝ SKRIPT PRO STUDENTY
# Skillmea · dataset: Vlnka Shop (čistá verze)
# =====================================================================================
# CO V TOMHLE SOUBORU JE
# Celý skript z hodiny: od načtení dat přes sezónnost, heatmapy, kohorty, rozdělení hodnot,
# korelace, model "koupí zákazník znovu?" až po mapu krajů a cvičení.
# Každý řádek má komentář, abyste si ho mohli doma projít a pochopit, co dělá.
#
# JAK SOUBOR POUŽÍT
#   1) Upravte cesty k CSV souborům (sekce 1) na svůj počítač – to je jediné, co musíte změnit.
#   2) Spouštějte skript po částech (po sekcích), ne celý najednou. Po každém grafu se podívejte,
#      co vidíte, a teprve pak jděte dál.
#   3) Čísla v titulcích grafů (např. "AUC 0,77", "−0,44") jsou do titulků napsaná ručně podle toho,
#      co vyšlo na našich datech. Když změníte data nebo nastavení, vždy je ověřte podle výpisu v konzoli.
#
# OBSAH
#   1  Načtení a příprava dat
#   2  Zákaznická tabulka a model (příprava pro pozdější grafy)
#   3  Sezónnost: malé čárové grafy a sezónní index
#   4  Heatmapy: den × hodina, dopravce × měsíc
#   5  Kohortní heatmapa retence
#   6  Rozdělení hodnot: histogram, boxplot, violin, ECDF
#   7  Vztahy mezi proměnnými: korelace, bublinový graf, hexbin
#   8  Hodnocení modelu: ROC, koeficienty, decily, matice záměn, prahy
#   9  Mapa krajů z čtverců
#  10  Špatné × lepší grafy (anti-vzory)
#  11  Cvičení
#
# SLOVNÍČEK ZKRATEK V KOMENTÁŘÍCH
#   fig = celé plátno (obrázek) · ax / osy = jeden graf (osa) uvnitř plátna · DataFrame = tabulka v pandas
#   Series = jeden sloupec tabulky · index = "popisky řádků" tabulky · train/test = data na učení / na ověření
# =====================================================================================

import pandas as pd                                          # pandas = práce s tabulkami (DataFrame)
import numpy as np                                           # numpy = matematika nad celými sloupci (log, odmocnina, pole)
import matplotlib.pyplot as plt                              # matplotlib = základní knihovna pro grafy
import matplotlib.colors as mcolors                          # nástroje pro práci s barvami (tady: převod hodnoty na barvu)
import seaborn as sns                                        # seaborn = hezčí a kratší grafy nad matplotlibem (heatmapa, boxplot, violin ...)

import os                                                    # práce se složkami a soubory
from sklearn.model_selection import train_test_split         # rozdělení dat na část pro učení a část pro test
from sklearn.linear_model import LogisticRegression          # logistická regrese = model pro předpověď ano/ne
from sklearn.metrics import roc_auc_score                    # AUC = jedno číslo: jak dobře model řadí zákazníky (0,5 = náhoda, 1 = ideál)
from sklearn.metrics import roc_curve                        # body ROC křivky (pro kreslení grafu)
from sklearn.metrics import confusion_matrix                 # matice záměn: kolikrát model trefil a kolikrát se spletl
os.makedirs("outputs", exist_ok=True)                        # vytvoří složku "outputs" pro obrázky; exist_ok=True = když už existuje, nic se nestane

# ---- barvy: definujeme je JEDNOU nahoře a dál používáme jen jejich názvy ----
MODRA="#2a78d6"                                              # hlavní barva grafů
SVETLA_MODRA="#a9c8f0"                                       # světlejší modrá (výplně boxů)
ORANZOVA="#eb6834"                                           # barva pro to, na co chceme upozornit
SEDA="#52514e"                                               # barva textů a popisků
SVETLE_SEDA="#c9c8c1"                                        # barva pro kontext, který nezvýrazňujeme

BARVY_SEGMENT={"Standard":MODRA, "Premium":ORANZOVA, "Firemní":"#2f8f6b"}   # slovník: segment zákazníka -> jeho barva (ve všech grafech stejná)
PORADI_SEGMENT=["Standard","Premium","Firemní"]              # pevné pořadí segmentů v grafech

plt.rcParams.update({"axes.spines.top":False, "axes.spines.right":False,   # rcParams = globální styl všech grafů; tady schováme horní a pravý rámeček
                     "axes.grid":True, "grid.color":"#e6e5e1", "grid.linewidth":0.8,   # zapneme světlou mřížku
                     "axes.axisbelow":True,                  # mřížka se kreslí POD data
                     "axes.edgecolor":"#c3c2b7", "axes.labelcolor":SEDA,   # barva os a popisků os
                     "xtick.color":SEDA, "ytick.color":SEDA, "font.size":10,   # barva a velikost čísel na osách
                     "axes.titleweight":"bold", "axes.titlesize":11})   # titulky grafů tučně a o něco větší

# =====================================================================================
# 1) NAČTENÍ A PŘÍPRAVA DAT
# =====================================================================================
# POZOR: cesty níže jsou z počítače lektora. Upravte je na složku, kam jste si CSV soubory uložili.
#        r"..." (raw string) znamená, že zpětná lomítka v cestě se berou doslova.
objednavky=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\objednavky2.csv")   # 1 řádek = 1 položka objednávky
produkty=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\produkty2.csv")       # katalog produktů (kategorie, cena za kus)
zakaznici=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\zakaznici2.csv")     # zákazníci (segment, kraj, zdroj)

objednavky["order_date"]=pd.to_datetime(objednavky["order_date"], format="%Y-%m-%d %H:%M")   # převod textu na pravé datum; format říká, jak text vypadá (rok-měsíc-den hodina:minuta)

op=objednavky.merge(produkty, on="product_id", how="left", validate="m:1")   # připojí k položkám informace o produktu; how="left" = žádnou položku neztratíme; validate="m:1" = pojistka, že každý produkt je v katalogu jen jednou
assert len(op)==len(objednavky)                              # KONTROLA: merge nesmí změnit počet řádků (kdyby ano, program spadne a víme, že je chyba)
op["trzba_bez_dph"]=op["quantity"]*op["cena_za_kus_bez_dph"]*(1-op["discount_pct"]/100)   # tržba položky = množství × cena × (1 − sleva v procentech/100)

agg_tabulka=(op                                              # závorky nám dovolí psát každou metodu na nový řádek
             .groupby("order_id")                            # rozdělí řádky do skupin: 1 skupina = 1 objednávka
             .agg(order_date=("order_date","first"),         # pro každou objednávku vezme první hodnotu data (u všech položek je stejná)
                  customer_id=("customer_id","first"),       # zákazník
                  status=("status","first"),                 # stav objednávky (dokonceno / vraceno / zruseno)
                  channel=("channel","first"),               # kanál (web / mobilni_app)
                  dopravce=("dopravce","first"),             # dopravce
                  delivery_days=("delivery_days","first"),   # doba doručení ve dnech
                  trzba_bez_dph=("trzba_bez_dph","sum"))     # tržba objednávky = SOUČET tržeb všech jejích položek
             .reset_index()                                  # order_id se vrátí z indexu zpět do normálního sloupce
             .sort_values(["order_date","order_id"]))        # seřadí podle data a pak podle čísla objednávky
print("Objednávek:", len(agg_tabulka))                       # KONTROLA: kolik je objednávek celkem
print(agg_tabulka.head())                                    # prvních 5 řádků tabulky

dok=agg_tabulka[agg_tabulka["status"]=="dokonceno"].copy()   # necháme jen dokončené objednávky; .copy() = samostatná kopie, se kterou můžeme bezpečně pracovat
dok=dok.merge(zakaznici[["customer_id","segment","kraj","zdroj"]], on="customer_id", how="left", validate="m:1")   # přidáme k objednávkám segment, kraj a zdroj zákazníka
assert len(dok)==23940                                       # KONTROLA: dokončených objednávek má být 23 940
dok["rok"]=dok["order_date"].dt.year                         # .dt = práce s datem: vytáhne rok
dok["mesic"]=dok["order_date"].dt.month                      # měsíc (1–12)
dok["rok_mesic"]=dok["order_date"].dt.to_period("M")         # období ve tvaru rok-měsíc (např. 2025-11); hodí se do grafů po měsících
dok["den_v_tydnu"]=dok["order_date"].dt.dayofweek            # den v týdnu jako číslo: pondělí = 0 ... neděle = 6
dok["hodina"]=dok["order_date"].dt.hour                      # hodina objednávky (0–23)

# =====================================================================================
# 2) ZÁKAZNICKÁ TABULKA A MODEL (z lekce Feature engineering)
# =====================================================================================
# Cíl modelu: předpovědět, jestli zákazník koupí znovu. Rozdělíme čas na HISTORII (z ní vyrobíme příznaky)
# a BUDOUCNOST (z ní zjistíme, jestli opravdu koupil). Model nesmí vidět budoucnost – jinak by "opisoval".
hranice=pd.Timestamp("2026-06-30 23:59")                     # "dnešek" z pohledu modelu: vše do tohoto data je historie
konec_okna=hranice+pd.Timedelta(days=123)                    # konec sledovaného okna v budoucnosti (+123 dní); Timedelta = časový rozdíl
obchody_historie=dok[dok["order_date"]<=hranice]             # objednávky z historie (z nich budou příznaky)
obchody_budoucnost=dok[(dok["order_date"]>hranice)&(dok["order_date"]<=konec_okna)]   # objednávky v budoucím okně (z nich bude cíl); & = "a zároveň", každá podmínka v závorce

zakaznik=pd.DataFrame(index=sorted(obchody_historie["customer_id"].unique()))   # prázdná tabulka, kde 1 řádek = 1 zákazník, který v historii nakoupil
zakaznik.index.name="customer_id"                            # pojmenujeme index
zakaznik["koupi_znovu"]=zakaznik.index.isin(obchody_budoucnost["customer_id"]).astype(int)   # CÍL: .isin = je zákazník v budoucích objednávkách? ano/ne; .astype(int) převede True/False na 1/0
print("Zákazníků:", len(zakaznik), "| podíl koupí znovu:", round(zakaznik["koupi_znovu"].mean(),3))   # průměr nul a jedniček = podíl těch, kdo koupili znovu

nakupy=(obchody_historie.groupby("customer_id")              # skupina = zákazník
        .agg(posledni_nakup=("order_date","max"),            # kdy nakoupil naposledy
             frekvence=("order_id","count"),                 # kolik měl objednávek
             monetary=("trzba_bez_dph","sum")))              # kolik celkem utratil
print(nakupy.head())                                         # kontrolní výpis
zakaznik=zakaznik.join(nakupy)                               # .join připojí tabulku podle indexu (customer_id)
zakaznik["recency_dny"]=(hranice-zakaznik["posledni_nakup"]).dt.days   # recency = kolik dní uplynulo od posledního nákupu
zakaznik["AVG_hodnota_objednavky"]=zakaznik["monetary"]/zakaznik["frekvence"]   # průměrná hodnota jedné objednávky
zakaznik["monetary_log"]=np.log1p(zakaznik["monetary"])      # logaritmus útraty (log1p = log(1+x)); zmenší vliv obřích hodnot

h=obchody_historie.sort_values(["customer_id","order_date"]).copy()   # seřadíme podle zákazníka a času (nutné, aby diff dával správný výsledek)
h["interval"]=h.groupby("customer_id")["order_date"].diff().dt.days   # .diff = rozdíl proti předchozímu řádku téhož zákazníka = dny mezi dvěma nákupy
zakaznik["interval_prumer"]=h.groupby("customer_id")["interval"].mean()   # průměrný odstup mezi nákupy; zákazník s jedním nákupem má NaN (prázdno)

posledni90=obchody_historie[obchody_historie["order_date"]>hranice-pd.Timedelta(days=90)]   # objednávky z posledních 90 dní před hranicí
zakaznik["objednavek_90d"]=posledni90.groupby("customer_id").size().reindex(zakaznik.index, fill_value=0)   # .size = počet řádků ve skupině; .reindex doplní 0 zákazníkům, kteří za 90 dní nenakoupili

vsechny=agg_tabulka[agg_tabulka["order_date"]<=hranice]      # všechny objednávky do hranice, i vrácené a zrušené
zakaznik["podil_vracenych"]=(vsechny["status"]!="dokonceno").groupby(vsechny["customer_id"]).mean().reindex(zakaznik.index)   # True/False "nebyla dokončená" průměrujeme po zákaznících = podíl vrácených/zrušených

segment=zakaznici.set_index("customer_id")["segment"].reindex(zakaznik.index)   # segment každého zákazníka ve stejném pořadí jako v tabulce zakaznik
zakaznik["segment_premium"]=(segment=="Premium").astype(int)   # one-hot: 1 = Premium, jinak 0
zakaznik["segment_firemni"]=(segment=="Firemní").astype(int)   # one-hot: 1 = Firemní (Standard = obě 0)

y=zakaznik["koupi_znovu"]                                    # y = to, co chceme předpovídat
X=zakaznik.drop(columns=["koupi_znovu","posledni_nakup","monetary"])   # X = příznaky: vše kromě cíle, surového data a surové útraty (máme její logaritmus)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)   # 80 % na učení, 20 % na test; random_state = stejné rozdělení při každém spuštění; stratify = stejný podíl "koupí znovu" v obou částech
median=X_train.median()                                      # medián každého sloupce, POUZE z trainu
X_train=X_train.fillna(median)                               # prázdné hodnoty (NaN) nahradíme mediánem
X_test=X_test.fillna(median)                                 # test doplníme čísly z TRAINU (test nesmí ovlivnit nic, co se model učí)
prumer=X_train.mean()                                        # průměr každého sloupce z trainu
odchylka=X_train.std()                                       # směrodatná odchylka z trainu
X_train_s=(X_train-prumer)/odchylka                          # standardizace (z-skóre): všechny příznaky mají srovnatelnou škálu
X_test_s=(X_test-prumer)/odchylka                            # test standardizujeme stejnými čísly z trainu

model=LogisticRegression(max_iter=1000)                      # vytvoříme model; max_iter = kolik pokusů smí dělat při hledání řešení
model.fit(X_train_s, y_train)                                # učení modelu: najde váhy příznaků (jen na trainu)
p_test=model.predict_proba(X_test_s)[:,1]                    # predict_proba vrací 2 sloupce (šance "ne" a "ano"); [:,1] vezme šanci "koupí znovu"
print("AUC modelu:", round(roc_auc_score(y_test, p_test),3))   # kvalita modelu na testu

X_vse=X.fillna(median)                                       # všichni zákazníci, chybějící hodnoty doplněné mediánem z trainu
zakaznik["pravdepodobnost"]=model.predict_proba((X_vse-prumer)/odchylka)[:,1]   # pravděpodobnost nákupu pro každého zákazníka
zakaznik["riziko"]=pd.cut(zakaznik["pravdepodobnost"], bins=[0,0.4,0.7,1], labels=["vysoké riziko","střední","věrný"])   # pd.cut rozdělí čísla do přihrádek: 0–0,4 / 0,4–0,7 / 0,7–1
zakaznik["segment"]=segment                                  # přidáme segment do tabulky zákazníků (pro grafy)
zakaznik["kraj"]=zakaznici.set_index("customer_id")["kraj"].reindex(zakaznik.index)   # kraj zákazníka
zakaznik["zdroj"]=zakaznici.set_index("customer_id")["zdroj"].reindex(zakaznik.index)   # odkud zákazník přišel

# =====================================================================================
# 3) SEZÓNNOST – malé čárové grafy (small multiples) a sezónní index
# =====================================================================================
# Pro sezónnost potřebujeme tržby po POLOŽKÁCH (ne po objednávkách), protože kategorie je vlastnost produktu.
radky=op[op["status"]=="dokonceno"].copy()                   # dokončené položky objednávek
radky["rok"]=radky["order_date"].dt.year                     # rok
radky["měsíc"]=radky["order_date"].dt.month                  # měsíc (s diakritikou – tento název používá pivot níže)
radky["mesic"]=radky["order_date"].dt.month                  # totéž bez diakritiky (kdybychom ho později potřebovali)
print(radky.info())                                          # .info() = přehled sloupců, typů a prázdných hodnot
kat_mesic=radky.pivot_table(index=["kategorie","měsíc"],     # řádky tabulky: dvojice kategorie + měsíc
                            columns="rok",                   # sloupce: roky
                            values="trzba_bez_dph",          # co se počítá
                            aggfunc="sum")/1e6               # součet tržeb; /1e6 = v milionech Kč
print(kat_mesic)                                             # kontrola

kat_poradi=(radky[radky["kategorie"]!="Vyřazené zboží"].groupby("kategorie")["trzba_bez_dph"].sum()   # tržby po kategoriích bez "Vyřazené zboží"
            .sort_values(ascending=False).index.to_list())   # seřadíme od největší a vezmeme jen názvy jako seznam
print(kat_poradi)                                            # pořadí kategorií podle tržeb

# ---- 3.1 Malé čárové grafy: každá kategorie ve svém políčku, rok 2025 šedě × rok 2026 modře ----
fig, osy=plt.subplots(2,3, figsize=(12,6))                   # mřížka 2 řádky × 3 sloupce grafů; figsize = velikost v palcích
osy=osy.flatten()                                            # mřížka 2×3 se "zploští" na seznam 6 os, ať můžeme procházet cyklem

for i, kat in enumerate(kat_poradi):                         # enumerate dává dvojici (pořadí i, název kategorie)
    ax=osy[i]                                                # i-tý graf z mřížky
    cast=kat_mesic.loc[kat]                                  # .loc[kat] vybere z pivotu jen řádky jedné kategorie (index = měsíc, sloupce = roky)
    ax.plot(cast.index, cast[2025], color=SVETLE_SEDA, label="2025")   # loňský rok šedě = kontext
    ax.plot(cast.index, cast[2026], color=MODRA, label="2026")         # letošní rok modře
    ax.set_title(kat, loc="left")                            # titulek = název kategorie, zarovnaný doleva
    ax.set_xticks(range(1,13))                               # na ose x ukážeme všech 12 měsíců
osy[0].legend(frameon=False)                                 # legenda jen v prvním grafu (stejné barvy platí všude); frameon=False = bez rámečku
fig.supxlabel("Měsíc v roce")                                # popisek osy x společný pro celé plátno
fig.supylabel("Tržby v milionech Kč")                        # popisek osy y společný pro celé plátno
fig.tight_layout()                                           # automaticky srovná mezery, aby se nic nepřekrývalo
fig.savefig("outputs/2026-16-11 male liniove grafy.png")     # uložení do souboru (VŽDY před plt.show(), jinak se uloží prázdný obrázek)
plt.show()                                                   # zobrazení grafu

# ---- 3.2 Sezónní index: každý měsíc dělíme průměrným měsícem (1 = průměr) ----
fig, osy=plt.subplots(2,3, figsize=(12,6))                   # opět 6 malých grafů
osy=osy.flatten()                                            # na seznam os
for i, kat in enumerate(kat_poradi):                         # pro každou kategorii
    ax = osy[i]                                              # její graf
    cast = kat_mesic.loc[kat][2025]                          # tržby kategorie v roce 2025 po měsících
    index=cast/cast.mean()                                   # sezónní index: měsíc ÷ průměr měsíců; 1,5 = o 50 % nad průměrem
    ax.axhline(1,color=SVETLE_SEDA, linestyle="--")          # vodorovná čárkovaná čára na hodnotě 1 = "průměrný měsíc"
    ax.plot(index.index, index.values, color=MODRA, marker="o")   # křivka indexu s kolečky
    ax.set_title(kat, loc="left")                            # název kategorie
    ax.set_xticks(range(1, 13))                              # 12 měsíců
osy[0].legend(frameon=False)                                 # (v tomto grafu nejsou pojmenované čáry, legenda zůstane prázdná; matplotlib může vypsat upozornění, nevadí)
fig.supxlabel("Měsíc v roce")                                # společný popisek osy x
fig.tight_layout()                                           # srovnání mezer
plt.show()                                                   # zobrazení

# ---- 3.3 Všechny kategorie v jednom grafu: šedě všechny, oranžově jedna (zvýraznění) ----
fig, ax=plt.subplots(figsize=(12,6))                         # jeden velký graf
for kat in kat_poradi:                                       # všechny kategorie
    rada=kat_mesic.loc[kat][2025]                            # tržby po měsících v roce 2025
    ax.plot(rada.index, rada/rada.mean(), color=SVETLE_SEDA) # sezónní index každé kategorie šedě
rada=kat_mesic.loc["Hračky"][2025]                           # nakonec vybereme Hračky...
ax.plot(rada.index, rada/rada.mean(), color=ORANZOVA, label="Hračky")   # ...a nakreslíme je jako POSLEDNÍ oranžově, aby ležely nahoře

ax.set_xlabel("Měsíc v roce")                                # popisek osy x
ax.set_ylabel("Sezonní index -> 1 průměrný měsíc")           # popisek osy y
ax.legend(frameon=False)                                     # legenda (jen "Hračky")
fig.tight_layout()                                           # srovnání mezer
plt.show()                                                   # zobrazení

# =====================================================================================
# 4) HEATMAPY – barva místo čísla
# =====================================================================================
# ---- 4.1 Kolik objednávek přijde v který den a hodinu ----
tab_dh=pd.crosstab(dok["den_v_tydnu"], dok["hodina"])        # crosstab = tabulka četností: řádky = den v týdnu, sloupce = hodina, v buňce počet objednávek
tab_dh.index=["Po","Út", "St", "Čt", "Pá", "So", "Ne"]       # přejmenujeme řádky z čísel 0–6 na zkratky dnů (pořadí odpovídá dayofweek)

# Varianta A: čistý matplotlib (imshow) – víc ručního nastavování
fig, ax=plt.subplots(figsize=(12,6))                         # jedno plátno
obrazek=ax.imshow(tab_dh.values,                             # imshow kreslí pole čísel jako barevné čtverečky; .values = čistá čísla bez popisků
                  cmap="Blues")                              # cmap = barevná škála (Blues = čím tmavší, tím víc)
ax.set_xticks(range(24))                                     # 24 značek na ose x (hodiny)
ax.set_xticklabels(tab_dh.columns)                           # popisky značek = čísla hodin
ax.set_yticks(range(7))                                      # 7 značek na ose y (dny)
ax.set_yticklabels(tab_dh.index)                             # popisky = zkratky dnů
fig.colorbar(obrazek, ax=ax)                                 # barevná legenda (pruh s hodnotami)
plt.show()                                                   # zobrazení

# Varianta B: seaborn – totéž na jeden příkaz, popisky se vezmou z tabulky samy
fig, ax=plt.subplots(figsize=(12,6))                         # jedno plátno
sns.heatmap(tab_dh,                                          # tabulka, kterou obarvíme
            cmap="Blues",                                    # barevná škála
            cbar_kws={"label": "Počet objednávek"},          # popisek barevné legendy
            ax=ax                                            # do kterého grafu kreslit
)
plt.show()                                                   # zobrazení

# ---- 4.2 Dva kanály vedle sebe (web × mobilní aplikace) ----
fig, osy=plt.subplots(1,2, figsize=(12,6))                   # 1 řádek, 2 grafy
for ax, kanal in zip(osy,["web", "mobilni_app"]):            # zip spáruje graf a kanál: (levý, "web"), (pravý, "mobilni_app")
    cast=dok[dok["channel"]==kanal]                          # jen objednávky daného kanálu
    tab=pd.crosstab(cast["den_v_tydnu"], cast["hodina"],normalize="all")*100   # normalize="all" = každá buňka je podíl z VŠECH objednávek kanálu; *100 = procenta
    tab.index = ["Po", "Út", "St", "Čt", "Pá", "So", "Ne"]   # názvy dnů
    sns.heatmap(tab,cmap="Blues",cbar_kws={"label": kanal}, ax=ax)   # heatmapa; popisek legendy = název kanálu
plt.show()                                                   # zobrazení

# ---- 4.3 Doba doručení: dopravce × měsíc ----
print(dok.info())                                            # přehled sloupců a typů
dop_mesic=dok.pivot_table(index="dopravce", columns="rok_mesic",values="delivery_days", aggfunc="mean")   # kontingenční tabulka: průměrná doba doručení pro každého dopravce a měsíc
dop_mesic.columns=dop_mesic.columns.astype(str)              # názvy sloupců (období) převedeme na text, jinak s nimi heatmapa může mít potíže

fig, ax=plt.subplots(figsize=(12,6))                         # plátno
sns.heatmap(dop_mesic,cmap="YlOrRd",                         # škála žlutá → červená: čím červenější, tím delší doručení
            annot=True,                                      # do každého čtverečku vypíše číslo
            fmt=".1f",                                       # formát čísla: 1 desetinné místo
            cbar_kws={"label": "Průměr dní doručení"}, ax=ax)   # popisek barevné legendy
ax.grid(False)                                               # u heatmapy vypneme mřížku z globálního stylu (rušila by)
plt.show()                                                   # zobrazení

# =====================================================================================
# 5) KOHORTNÍ HEATMAPA RETENCE
# =====================================================================================
# Kohorta = skupina zákazníků, kteří poprvé nakoupili ve stejném měsíci. Sledujeme, kolik % z nich nakoupí v dalších měsících.
dok["mesic_cislo"]=dok["order_date"].dt.year*12+dok["order_date"].dt.month   # měsíc jako jedno pořadové číslo (rok×12+měsíc) – díky tomu se měsíce jednoduše odečítají
prvni_cislo=dok.groupby("customer_id")["mesic_cislo"].min()  # pro každého zákazníka číslo měsíce jeho prvního nákupu
prvni_text=dok.groupby("customer_id")["order_date"].min().dt.strftime("%Y-%m")   # totéž jako text "2025-11" (štítek kohorty); strftime = formátování data na text
dok["kohorta"]=dok["customer_id"].map(prvni_text)            # .map ke každé objednávce doplní kohortu jejího zákazníka
dok["mesic_od_prvniho"]=dok["mesic_cislo"]-dok["customer_id"].map(prvni_cislo)   # kolik měsíců uplynulo od prvního nákupu (0 = první měsíc, 1 = další ...)
kohorty=dok.groupby(["kohorta","mesic_od_prvniho"])["customer_id"].nunique().unstack()   # počet RŮZNÝCH zákazníků (nunique) v každé kohortě a měsíci; .unstack přetočí měsíce do sloupců
retence=kohorty.div(kohorty[0], axis=0)                      # podíl = počet ÷ velikost kohorty (sloupec 0); axis=0 = dělení po řádcích
retence_graf=retence.drop(index="2025-01")                   # vyřadíme kohortu 2025-01: nejsou to nováčci, ale všichni, kdo existovali na začátku dat
retence_graf=retence_graf.iloc[:, 1:13].dropna(how="all")    # vezmeme měsíce 1–12 po prvním nákupu (sloupec 0 je vždy 100 %); dropna(how="all") zahodí řádky, které jsou celé prázdné
retence_graf.index=retence_graf.index+" (n="+kohorty.loc[retence_graf.index,0].astype(int).astype(str)+")"   # k názvu kohorty připojíme velikost: "2025-03 (n=150)"

fig, ax=plt.subplots(figsize=(12,7.5))                       # plátno
sns.heatmap(retence_graf, cmap="Blues", vmin=0, vmax=0.55, annot=True, fmt=".0%", annot_kws={"size":7},   # vmin/vmax = rozsah barevné škály; fmt=".0%" = čísla jako procenta; annot_kws = vzhled čísel
            linewidths=0.5, cbar_kws={"label":"Podíl kohorty, který v daném měsíci nakoupil"}, ax=ax)   # linewidths = tenké bílé čáry mezi čtverečky
ax.grid(False)                                               # vypnout mřížku
ax.set_xlabel("Měsíců od prvního nákupu"); ax.set_ylabel("Kohorta (měsíc prvního nákupu)")   # popisky os (dva příkazy na jednom řádku odděleny středníkem)
ax.set_title("V každém dalším měsíci nakoupí zhruba třetina nových zákazníků; kohorty se příliš neliší", loc="left")   # titulek = závěr, který z grafu čteme
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_08_kohortni_heatmapa.png", dpi=120)  # uložení; dpi = rozlišení
plt.show()                                                   # zobrazení
print(retence_graf.mean().round(2))                          # KONTROLA: průměrná retence v každém měsíci po prvním nákupu

# =====================================================================================
# 6) ROZDĚLENÍ HODNOT – histogram, boxplot, violin, ECDF
# =====================================================================================
# ---- 6.1 Histogram: lineární × logaritmická osa ----
fig, osy=plt.subplots(1, 2, figsize=(12,4))                  # 2 grafy vedle sebe
osy[0].hist(dok["trzba_bez_dph"], bins=50, color=MODRA, edgecolor="white")   # histogram: rozdělí hodnoty do 50 "košů" a spočítá, kolik objednávek je v každém
osy[0].set_title("Lineární osa: vidíme jen nejmenší objednávky", loc="left")   # většina objednávek je malá, pár obřích natáhne osu
hranice_binu=np.logspace(np.log10(dok["trzba_bez_dph"].min()), np.log10(dok["trzba_bez_dph"].max()), 50)   # np.logspace = 50 hranic košů rovnoměrně rozložených v logaritmu (koše se zvětšují)
osy[1].hist(dok["trzba_bez_dph"], bins=hranice_binu, color=MODRA, edgecolor="white")   # histogram s našimi logaritmickými koši
osy[1].set_xscale("log")                                     # osa x v logaritmické škále: stejná vzdálenost = stejný NÁSOBEK
osy[1].set_title("Logaritmická osa: vidíme celé rozdělení", loc="left")   # teď je vidět celý tvar rozdělení
for ax in osy:                                               # popisky nastavíme obou grafům najednou
    ax.set_xlabel("Hodnota objednávky (Kč bez DPH)"); ax.set_ylabel("Počet objednávek")   # popisky os
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_09_histogram_log.png", dpi=120)      # uložení
plt.show()                                                   # zobrazení

# ---- 6.2 Boxplot podle segmentu ----
fig, ax=plt.subplots(figsize=(8,5))                          # plátno
sns.boxplot(data=dok, x="segment", y="trzba_bez_dph", order=PORADI_SEGMENT,   # box = prostřední polovina hodnot, čára uvnitř = medián, "vousy" a tečky = okraje a odlehlé hodnoty
            color=SVETLA_MODRA, fliersize=2, ax=ax)          # fliersize = velikost teček odlehlých hodnot
ax.set_yscale("log")                                         # osa y logaritmicky (hodnoty se liší o řády)
prumery=dok.groupby("segment")["trzba_bez_dph"].mean().reindex(PORADI_SEGMENT)   # průměr hodnoty objednávky v každém segmentu, ve správném pořadí
ax.scatter(range(3), prumery.values, marker="D", color=ORANZOVA, s=60, zorder=3, label="průměr")   # průměry jako oranžové kosočtverce; zorder=3 = nakreslit nad box; s = velikost
ax.legend(frameon=False, loc="upper left")                   # legenda vlevo nahoře
ax.set_xlabel(""); ax.set_ylabel("Hodnota objednávky (Kč bez DPH, log osa)")   # prázdný popisek x (kategorie mluví samy za sebe)
ax.set_title("Firemní objednávky jsou typicky 4× větší; průměr (◆) tahá nahoru dlouhý ocas", loc="left")   # závěr; všimněte si rozdílu medián × průměr
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_10_boxplot_segment.png", dpi=120)    # uložení
plt.show()                                                   # zobrazení
print(dok.groupby("segment")["trzba_bez_dph"].agg(["median","mean"]).round(0))   # KONTROLA: medián a průměr podle segmentu

# ---- 6.3 Violin: tvar rozdělení recency podle toho, jestli zákazník koupil znovu ----
zakaznik["koupil_text"]=np.where(zakaznik["koupi_znovu"]==1, "koupil znovu", "nekoupil")   # np.where(podmínka, když ano, když ne) = z 1/0 udělá srozumitelný text
fig, ax=plt.subplots(figsize=(7,5))                          # plátno
sns.violinplot(data=zakaznik, x="koupil_text", y="recency_dny",   # violin = "zrcadlový histogram": čím širší místo, tím víc zákazníků
               order=["koupil znovu","nekoupil"],            # pořadí houslí
               inner="quartile",                             # uvnitř čárky pro kvartily (25 %, medián, 75 %)
               cut=0,                                        # housle neprodlužujeme za nejmenší a největší skutečnou hodnotu
               palette={"koupil znovu":MODRA, "nekoupil":ORANZOVA},   # barvy podle skupiny
               hue="koupil_text", legend=False,              # hue = obarvení podle skupiny; legenda zbytečná, popisky jsou pod houslemi
               ax=ax)
ax.set_xlabel(""); ax.set_ylabel("Dny od posledního nákupu (recency)")   # popisky os
ax.set_title("Kdo koupil znovu, nakoupil nedávno", loc="left")   # závěr
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_11_violin_recency.png", dpi=120)     # uložení
plt.show()                                                   # zobrazení

# ---- 6.4 ECDF: kumulativní rozdělení (z grafu se dají přímo číst mediány a kvartily) ----
fig, ax=plt.subplots(figsize=(8,4.5))                        # plátno
sns.ecdfplot(data=dok, x="trzba_bez_dph", hue="segment",     # ECDF: pro každou hodnotu ukáže podíl objednávek, které jsou menší nebo rovny; jedna čára za segment
             hue_order=PORADI_SEGMENT, palette=BARVY_SEGMENT,   # pořadí a barvy segmentů
             log_scale=True, linewidth=2.5, ax=ax)           # log_scale = logaritmická osa x; linewidth = tloušťka čáry
ax.axhline(0.5, color=SEDA, linewidth=1, linestyle="--")     # čára na 50 % = medián (kde ji čára segmentu protne, tam je medián)
ax.axhline(0.25, color=SEDA, linewidth=1, linestyle="--")    # čára na 25 % = dolní kvartil
ax.set_xlabel("Hodnota objednávky (Kč bez DPH, log osa)"); ax.set_ylabel("Podíl objednávek menších než hodnota")   # popisky os
ax.set_title("Čára Firemních je posunutá doprava: polovina jejich objednávek přesáhne 8 600 Kč", loc="left")   # závěr (číslo je zapsané ručně, ověřte ho)
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_12_ecdf_segment.png", dpi=120)       # uložení
plt.show()                                                   # zobrazení

# =====================================================================================
# 7) VZTAHY MEZI PROMĚNNÝMI – korelace, bublinový graf, hexbin
# =====================================================================================
# ---- 7.1 Korelační matice ----
# Korelace je číslo od −1 do +1: +1 = dvě veličiny rostou spolu, −1 = jedna roste a druhá klesá, 0 = žádný lineární vztah.
sloupce=["koupi_znovu","recency_dny","frekvence","monetary_log","AVG_hodnota_objednavky","interval_prumer","objednavek_90d","podil_vracenych"]   # které sloupce porovnáváme
kor=zakaznik[sloupce].corr()                                 # .corr() spočítá korelaci každého sloupce s každým
maska=np.triu(np.ones_like(kor, dtype=bool), k=1)            # maska pro horní trojúhelník nad úhlopříčkou (matice je souměrná, polovina by byla zbytečná)
fig, ax=plt.subplots(figsize=(9,6.5))                        # plátno
sns.heatmap(kor, mask=maska,                                 # mask = tyto čtverečky se nevykreslí
            cmap="RdBu_r",                                   # divergující škála: červená = kladná, modrá = záporná, bílá = nula
            vmin=-1, vmax=1, center=0,                       # rozsah škály −1 až +1 a bílá uprostřed
            annot=True, fmt=".2f", annot_kws={"size":8},     # čísla v buňkách na 2 desetinná místa
            square=True,                                     # čtvercové buňky
            linewidths=0.5, cbar_kws={"shrink":0.8, "label":"korelace"}, ax=ax)   # shrink = menší barevná legenda
ax.grid(False)                                               # vypnout mřížku
ax.set_title("Kdo dlouho nenakoupil, ten se nevrací: koupi_znovu × recency = −0,44", loc="left")   # závěr s číslem z matice
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_13_korelace.png", dpi=120)           # uložení
plt.show()                                                   # zobrazení

# ---- 7.2 Bublinový graf: čtyři údaje v jednom obrázku ----
fig, ax=plt.subplots(figsize=(9,5.5))                        # plátno
bubliny=ax.scatter(zakaznik["recency_dny"], zakaznik["frekvence"],   # x = recency, y = frekvence; každý zákazník = jedna bublina
                   s=np.sqrt(zakaznik["monetary"])/6,        # s = velikost bubliny podle útraty; odmocnina zmírní rozdíly, /6 jen ladí velikost
                   c=zakaznik["pravdepodobnost"],            # c = barva bubliny podle pravděpodobnosti nákupu z modelu
                   cmap="RdYlBu", vmin=0, vmax=1,            # škála červená–žlutá–modrá od 0 do 1
                   alpha=0.55, edgecolor="none")             # alpha = průhlednost (překrývající se bubliny je vidět); bez obrysu
fig.colorbar(bubliny, ax=ax, label="Pravděpodobnost nákupu (model)")   # barevná legenda
ax.set_xlabel("Dny od posledního nákupu (recency)"); ax.set_ylabel("Počet objednávek (frekvence)")   # popisky os
ax.set_title("Nedávno a často = šance blízko 100 %; dlouho a málo = téměř ztracený zákazník", loc="left")   # závěr
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_14_bubliny.png", dpi=120)            # uložení
plt.show()                                                   # zobrazení

# ---- 7.3 Hexbin: když je bodů moc, spočítáme průměr v šestiúhelnících ----
fig, ax=plt.subplots(figsize=(9,5.5))                        # plátno
hexy=ax.hexbin(zakaznik["recency_dny"], zakaznik["frekvence"],   # rovinu rozdělí na šestiúhelníky
               C=zakaznik["koupi_znovu"],                    # C = hodnota, kterou v každém šestiúhelníku sečteme
               reduce_C_function=np.mean,                    # sečtení = průměr; průměr nul a jedniček = podíl těch, kdo koupili znovu
               gridsize=18,                                  # kolik šestiúhelníků vedle sebe
               mincnt=8,                                     # šestiúhelník s méně než 8 zákazníky nekreslit (průměr z pár lidí je nespolehlivý)
               cmap="Blues", vmin=0, vmax=1)                 # škála 0–100 %
fig.colorbar(hexy, ax=ax, label="Podíl, kdo koupil znovu")   # barevná legenda
ax.set_xlabel("Dny od posledního nákupu (recency)"); ax.set_ylabel("Počet objednávek (frekvence)")   # popisky os
ax.grid(False)                                               # vypnout mřížku
ax.set_title("Mapa šancí: vlevo nahoře je koupě skoro jistá, vpravo dole skoro nikdo nekoupí", loc="left")   # závěr
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_15_hexbin.png", dpi=120)             # uložení
plt.show()                                                   # zobrazení

# =====================================================================================
# 8) HODNOCENÍ MODELU – ROC, koeficienty, decily, matice záměn, prahy
# =====================================================================================
# ---- 8.1 ROC křivka: porovnáme tři sady příznaků ----
# ROC ukazuje kompromis: kolik skutečných kupců model najde (TPR) × kolik nekupců omylem označí za kupce (FPR).
# Čím víc je křivka nahoře vlevo, tím lepší model. Úhlopříčka = hádání náhodou.
sady={"A: jen recency":["recency_dny"],                      # slovník: název sady -> seznam sloupců, které model smí použít
      "B: recency + frekvence + útrata":["recency_dny","frekvence","monetary_log"],
      "C: všechny příznaky":list(X_train_s.columns)}         # C = všechny sloupce tabulky X
barvy_sad=["#9a9892", "#7fb0ec", MODRA]                      # od nejslabší sady po nejsilnější

fig, ax=plt.subplots(figsize=(6.5,6.5))                      # čtvercové plátno
ax.plot([0,1], [0,1], color=SEDA, linestyle="--", linewidth=1, label="náhoda (AUC 0,5)")   # úhlopříčka = náhodný model
for (nazev, sloupce_sady), barva in zip(sady.items(), barvy_sad):   # cyklus přes sady; .items() dává dvojice (název, seznam sloupců)
    m=LogisticRegression(max_iter=1000)                      # nový model pro tuto sadu
    m.fit(X_train_s[sloupce_sady], y_train)                  # učení jen na vybraných sloupcích
    p=m.predict_proba(X_test_s[sloupce_sady])[:,1]           # pravděpodobnosti na testu
    fpr, tpr, prahy=roc_curve(y_test, p)                     # roc_curve vrací souřadnice křivky (FPR, TPR) pro všechny možné prahy
    auc_sady=roc_auc_score(y_test, p)                        # plocha pod křivkou = AUC
    ax.plot(fpr, tpr, color=barva, linewidth=2.5, label=f"{nazev} (AUC {auc_sady:.3f})")   # křivka; f-string vloží AUC do legendy na 3 desetinná místa
    print(nazev, "AUC =", round(auc_sady,3))                 # KONTROLA: AUC každé sady
tip=(p_test>=0.5)                                            # tip plného modelu při prahu 0,5: True = "koupí"
tpr_05=(tip & (y_test==1)).sum()/(y_test==1).sum()           # TPR: podíl skutečných kupců, které model označil za kupce
fpr_05=(tip & (y_test==0)).sum()/(y_test==0).sum()           # FPR: podíl nekupců, které model omylem označil za kupce
ax.scatter([fpr_05], [tpr_05], color=ORANZOVA, s=70, zorder=3)   # oranžový bod = kde model leží při prahu 0,5
ax.annotate("práh 0,5", (fpr_05, tpr_05), xytext=(fpr_05+0.08, tpr_05-0.10), color=ORANZOVA, arrowprops={"arrowstyle":"->","color":ORANZOVA})   # text se šipkou na ten bod (xytext = kde je text)
ax.set_xlabel("FPR: podíl nekupců, které model omylem označil za kupce"); ax.set_ylabel("TPR: podíl kupců, které model správně našel")   # popisky os
ax.set_aspect("equal")                                       # stejné měřítko osy x i y (čtverec)
ax.legend(frameon=False, loc="lower right")                  # legenda vpravo dole (tam je volno)
ax.set_title("Samotná recency dá AUC 0,77; další příznaky ho zvednou jen o 0,02", loc="left")   # závěr (čísla ověřte podle výpisu)
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_16_roc.png", dpi=120)                # uložení
plt.show()                                                   # zobrazení

# ---- 8.2 Koeficienty: které příznaky model považuje za nejdůležitější ----
koef=pd.Series(model.coef_[0], index=X_train_s.columns).sort_values()   # model.coef_ = naučené váhy; spojíme je s názvy příznaků a seřadíme od nejnižší
barvy=np.where(koef>0, MODRA, ORANZOVA)                      # kladná váha modře, záporná oranžově
fig, ax=plt.subplots(figsize=(8,5))                          # plátno
ax.barh(koef.index, koef.values, color=barvy, height=0.65)   # barh = vodorovné sloupce
ax.axvline(0, color=SEDA, linewidth=1)                       # svislá čára na nule
ax.grid(axis="y", visible=False)                             # vodorovnou mřížku vypneme
ax.set_xlabel("Váha příznaku v modelu (příznaky jsou standardizované, váhy jsou srovnatelné)")   # proč lze váhy porovnávat: standardizace
nejvetsi=koef.abs().sort_values(ascending=False).index[0]    # název příznaku s největší absolutní vahou (.abs() = bez znaménka)
ax.set_title(f"Nejsilnější příznak: {nejvetsi}\nmodrá = zvyšuje šanci na nákup, oranžová = snižuje", loc="left")   # \n = nový řádek v titulku
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_17_koeficienty.png", dpi=120)        # uložení
plt.show()                                                   # zobrazení
print(koef.round(2))                                         # KONTROLA: váhy příznaků

# ---- 8.3 Decily: seřadíme zákazníky podle pravděpodobnosti a rozdělíme na 10 stejných skupin ----
test=pd.DataFrame({"skutecnost":y_test.values, "p":p_test})  # tabulka: skutečnost (1/0) a pravděpodobnost z modelu
test["decil"]=pd.qcut(test["p"].rank(method="first"), 10, labels=False)   # qcut = 10 skupin se stejným počtem zákazníků; rank(method="first") zajistí, že hranice skupin jsou jedinečné; labels=False = skupiny 0–9
decily=test.groupby("decil").agg(podil_kupcu=("skutecnost","mean"),   # skutečný podíl kupců v decilu
                                 p_prumer=("p","mean"),      # průměrná pravděpodobnost, kterou model tipoval
                                 pocet=("skutecnost","size"))   # počet zákazníků v decilu
print(decily.round(2))                                       # KONTROLA: tabulka decilů

fig, osy=plt.subplots(1, 2, figsize=(13,4.8))                # dva grafy vedle sebe
# levý graf: sloupce = skutečnost, čára = co tipoval model (když se shodují, model je dobře kalibrovaný)
osy[0].bar(decily.index+1, decily["podil_kupcu"], color=MODRA, width=0.7, label="skutečně koupili")   # +1, ať decily jsou 1–10 místo 0–9
osy[0].plot(decily.index+1, decily["p_prumer"], color=ORANZOVA, marker="o", linewidth=2, label="co tipoval model")   # tip modelu
osy[0].axhline(y_test.mean(), color=SEDA, linestyle="--", linewidth=1)   # celkový podíl kupců = srovnávací úroveň
osy[0].text(0.55, y_test.mean()+0.02, "celkový průměr", color=SEDA, ha="left", va="bottom", fontsize=9)   # popisek čáry přímo v grafu; ha/va = zarovnání textu
osy[0].set_xticks(range(1,11))                               # decily 1–10
osy[0].set_xlabel("Decil zákazníků podle pravděpodobnosti (1 = nejnižší, 10 = nejvyšší)"); osy[0].set_ylabel("Podíl, kdo koupil znovu")   # popisky os
osy[0].set_ylim(0,1.05)                                      # osa y 0–105 %
osy[0].legend(frameon=False, loc="upper left")               # legenda
osy[0].set_title(f"V nejlepším decilu koupí znovu {decily['podil_kupcu'].iloc[-1]:.0%}, v nejhorším jen {decily['podil_kupcu'].iloc[0]:.0%}", loc="left")   # .iloc[-1] = poslední decil, .iloc[0] = první; titulek se počítá z dat
# pravý graf: gain křivka – kolik odcházejících zachytíme, když osloovíme nejrizikovější část zákazníků
poradi_zakazniku=test.sort_values("p")                       # od nejnižší pravděpodobnosti nákupu (= nejrizikovější)
odchod=(poradi_zakazniku["skutecnost"]==0).astype(int)       # 1 = zákazník skutečně neodešel k nákupu = "odchod"
zisk=odchod.cumsum().values/odchod.sum()                     # kumulativní součet ÷ celkový počet odchodů = podíl odchodů zachycených do dané pozice
osloveno=np.arange(1, len(poradi_zakazniku)+1)/len(poradi_zakazniku)   # podíl oslovených zákazníků (od 0 do 1)
osy[1].plot(osloveno, zisk, color=MODRA, linewidth=2.5, label="podle modelu")   # křivka modelu
osy[1].plot([0,1], [0,1], color=SEDA, linestyle="--", linewidth=1, label="náhodný výběr")   # úhlopříčka = oslovování náhodně
bod=int(0.2*len(poradi_zakazniku))                           # pozice, která odpovídá nejrizikovější pětině
osy[1].scatter([osloveno[bod-1]], [zisk[bod-1]], color=ORANZOVA, s=70, zorder=3)   # zvýrazněný bod
osy[1].annotate(f"20 % oslovených = {zisk[bod-1]:.0%} odcházejících", (osloveno[bod-1], zisk[bod-1]),
                xytext=(0.28, 0.12), color=ORANZOVA, arrowprops={"arrowstyle":"->","color":ORANZOVA})   # text se šipkou; hodnota se počítá z dat
osy[1].set_xlabel("Podíl nejrizikovějších zákazníků, které osloví obchodník"); osy[1].set_ylabel("Podíl odcházejících, které zachytí")   # popisky os
osy[1].legend(frameon=False, loc="lower right")              # legenda
osy[1].set_title(f"Oslovení nejrizikovější pětiny zachytí {zisk[bod-1]:.0%} odchodů (náhodně 20 %)", loc="left")   # závěr
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_18_decily_gain.png", dpi=120)        # uložení
plt.show()                                                   # zobrazení

# ---- 8.4 Matice záměn při prahu 0,5 ----
cm=confusion_matrix(y_test, (p_test>=0.5).astype(int))       # tabulka 2×2: skutečnost × tip modelu; (p>=0.5).astype(int) = tip 1/0
fig, ax=plt.subplots(figsize=(5.5,4.5))                      # plátno
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,   # fmt="d" = celá čísla; cbar=False = bez barevné legendy
            annot_kws={"size":14}, linewidths=1,             # velká čísla a tenké čáry mezi poli
            xticklabels=["tip: nekoupí","tip: koupí"],       # popisky sloupců = co model řekl
            yticklabels=["skutečně nekoupil","skutečně koupil"],   # popisky řádků = jak to bylo doopravdy
            ax=ax)
ax.grid(False)                                               # vypnout mřížku
ax.set_title("Matice záměn při prahu 0,5", loc="left")       # titulek
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_19_matice_zamen.png", dpi=120)       # uložení
plt.show()                                                   # zobrazení
tn, fp, fn, tp=cm.ravel()                                    # .ravel() srovná matici 2×2 do řady a rozbalí do čtyř proměnných v tomto pořadí: tn, fp, fn, tp
print("Správně nekoupí:", tn, "| omylem koupí:", fp, "| omylem nekoupí:", fn, "| správně koupí:", tp)   # tn = pravdivě negativní, fp = falešně pozitivní, fn = falešně negativní, tp = pravdivě pozitivní

# ---- 8.5 Jak se mění výsledek, když posuneme práh ----
print("práh | označeno jako rizikoví | z nich skutečně odešlo (přesnost) | z odcházejících zachyceno")   # hlavička tabulky ve výpisu
for prah in [0.3, 0.4, 0.5, 0.6, 0.7]:                       # zkusíme pět různých prahů
    riziko_tip=(p_test<prah)                                 # True = model zákazníka označil za rizikového (pravděpodobnost pod prahem)
    zachyceno=(riziko_tip & (y_test.values==0)).sum()        # kolik z označených skutečně nenakoupilo (správný zásah)
    print(f"{prah:.1f}  | {riziko_tip.sum():>5}                  | {zachyceno/max(riziko_tip.sum(),1):>6.0%}                           | {zachyceno/(y_test.values==0).sum():>6.0%}")   # přesnost = zásahy ÷ označení; zachyceno = zásahy ÷ všichni odcházející; max(...,1) chrání před dělením nulou; :>5 = zarovnání doprava
# Čím vyšší práh, tím víc zákazníků označíme (zachytíme víc odcházejících), ale tím víc označíme zbytečně (klesá přesnost).

# =====================================================================================
# 9) MAPA KRAJŮ Z ČTVERCŮ (tile map)
# =====================================================================================
# Každý kraj je jeden čtverec na přibližné poloze na mapě. Je to čitelnější než skutečná mapa, kde velké kraje zabírají víc místa než ty malé.
poloha_kraju={"Karlovarský":(0,0), "Ústecký":(1,0), "Liberecký":(2,0), "Královéhradecký":(3,0),   # slovník: kraj -> (sloupec, řádek) v mřížce
              "Plzeňský":(0,1), "Středočeský":(1,1), "Praha":(2,1), "Pardubický":(3,1), "Olomoucký":(4,1), "Moravskoslezský":(5,1),
              "Jihočeský":(1,2), "Vysočina":(3,2), "Jihomoravský":(4,2), "Zlínský":(5,2)}

kraje=dok.groupby("kraj").agg(trzba=("trzba_bez_dph","sum"),   # tržba kraje
                              zakazniku=("customer_id","nunique"))   # počet různých zákazníků v kraji
kraje["trzba_mil"]=kraje["trzba"]/1e6                        # tržba v milionech Kč
kraje["trzba_na_zak"]=kraje["trzba"]/kraje["zakazniku"]/1e3  # tržba na jednoho zákazníka v tisících Kč
print(kraje.sort_values("trzba_mil", ascending=False).round(1))   # KONTROLA: kraje od největšího

fig, osy=plt.subplots(1, 2, figsize=(13,4.2))                # dvě mapy vedle sebe
metriky_kraju=[("trzba_mil","Tržba kraje (mil. Kč)",plt.cm.Blues), ("trzba_na_zak","Tržba na zákazníka (tis. Kč)",plt.cm.Greens)]   # (sloupec, titulek, barevná paleta)
for ax, (sloupec, titulek, paleta) in zip(osy, metriky_kraju):   # pro každou mapu jedna metrika
    hodnoty=kraje[sloupec]                                   # hodnoty metriky pro všechny kraje
    norm=mcolors.Normalize(vmin=hodnoty.min(), vmax=hodnoty.max())   # Normalize převede hodnotu na číslo 0–1 (nejmenší = 0, největší = 1)
    for kraj, (sl, rad) in poloha_kraju.items():             # pro každý kraj: jeho název a poloha (sloupec, řádek)
        hodnota=hodnoty[kraj]                                # hodnota tohoto kraje
        barva_pole=paleta(0.15+0.85*norm(hodnota))           # barva z palety; 0,15+... zajistí, že ani nejmenší kraj není úplně bílý
        ax.add_patch(plt.Rectangle((sl, -rad), 0.94, 0.94, facecolor=barva_pole, edgecolor="white"))   # nakreslí čtverec; -rad = první řádek nahoře; 0,94 = rozměr, zbytek je mezera
        barva_textu="white" if norm(hodnota)>0.55 else SEDA  # na tmavém čtverci bílý text, na světlém šedý (čitelnost)
        ax.text(sl+0.47, -rad+0.55, kraj, ha="center", va="center", fontsize=7.5, color=barva_textu)   # název kraje uprostřed čtverce
        ax.text(sl+0.47, -rad+0.25, f"{hodnota:.1f}", ha="center", va="center", fontsize=10, fontweight="bold", color=barva_textu)   # číslo pod názvem
    ax.set_xlim(-0.1, 6.1); ax.set_ylim(-2.1, 1.05)          # rozsah os, aby se vešly všechny čtverce
    ax.set_aspect("equal"); ax.axis("off")                   # čtverce zůstanou čtvercové; osy a rámeček se schovají
    ax.set_title(titulek, loc="left")                        # titulek mapy
fig.suptitle("Absolutně vede Praha, ale na zákazníka dobře vychází i Zlínský kraj: velikost kraje a hodnota zákazníka nejsou totéž", x=0.01, ha="left", fontsize=12, fontweight="bold")   # společný titulek přes obě mapy
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_20_mapa_kraju.png", dpi=120)         # uložení
plt.show()                                                   # zobrazení

# =====================================================================================
# 10) ŠPATNĚ × LÉPE – tři nejčastější chyby v grafech
# =====================================================================================
kat=radky.groupby("kategorie")["trzba_bez_dph"].sum().sort_values(ascending=False)/1e6   # tržby po kategoriích v milionech Kč (včetně "Vyřazené zboží" – proto 7 výsečí)
podil_app=(dok["channel"]=="mobilni_app").groupby(dok["rok"]).mean()   # podíl objednávek z mobilní aplikace v každém roce (průměr z True/False)

fig, osy=plt.subplots(2, 3, figsize=(14,8))                  # horní řádek = špatně, dolní řádek = lépe
# Chyba 1: koláč se sedmi výsečemi
osy[0,0].pie(kat.values, labels=kat.index, textprops={"fontsize":8})   # koláčový graf; oko špatně porovnává úhly a plochy
osy[0,0].set_title("ŠPATNĚ: koláč se 7 výsečemi", loc="left", color=ORANZOVA)
osy[1,0].barh(kat.index[::-1], kat.values[::-1], color=MODRA, height=0.6)   # [::-1] otočí pořadí, aby největší bylo nahoře
for i, v in enumerate(kat.values[::-1]):                     # u každého sloupce napíšeme jeho hodnotu
    osy[1,0].text(v, i, f" {v:.1f}", va="center", color=SEDA, fontsize=9)
osy[1,0].grid(axis="y", visible=False)                       # vodorovnou mřížku vypneme
osy[1,0].set_title("LÉPE: seřazené sloupce s čísly", loc="left", color=MODRA)
# Chyba 2: osa y, která nezačíná v nule
osy[0,1].bar(["2025","2026"], podil_app.values, color=MODRA, width=0.5)
osy[0,1].set_ylim(0.20, 0.32)                                # ořezaná osa y – malý rozdíl vypadá obrovský
osy[0,1].set_title("ŠPATNĚ: osa y od 0,20 – rozdíl vypadá 4×", loc="left", color=ORANZOVA)
osy[1,1].bar(["2025","2026"], podil_app.values, color=MODRA, width=0.5)
osy[1,1].set_ylim(0, 0.4)                                    # osa od nuly – poměr sloupců odpovídá poměru hodnot
for i, v in enumerate(podil_app.values):                     # čísla nad sloupci
    osy[1,1].text(i, v+0.01, f"{v:.0%}", ha="center", color=SEDA, fontweight="bold")
osy[1,1].set_title("LÉPE: osa od nuly – růst 1,4× je vidět, ale nepřehnaný", loc="left", color=MODRA)
osy[1,1].set_ylabel("Podíl objednávek z mobilní aplikace")
# Chyba 3: duhová paleta
osy[0,2].imshow(tab_dh.values, cmap="jet", aspect="auto")    # "jet" = duha; aspect="auto" roztáhne heatmapu na plochu grafu
osy[0,2].set_title("ŠPATNĚ: duha – je žlutá 'víc' než zelená?", loc="left", color=ORANZOVA)
osy[1,2].imshow(tab_dh.values, cmap="Blues", aspect="auto")  # jedna barva, tmavší = víc
osy[1,2].set_title("LÉPE: jedna barva, tmavší = víc", loc="left", color=MODRA)
for ax in [osy[0,2], osy[1,2], osy[0,0]]:                    # u těchto tří grafů vypneme mřížku
    ax.grid(False)
fig.suptitle("Stejná data, jiný dojem: tři nejčastější chyby", x=0.01, ha="left", fontsize=14, fontweight="bold")   # společný titulek
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_21_anti_vzory.png", dpi=120)         # uložení
plt.show()                                                   # zobrazení
print(podil_app.round(3))                                    # KONTROLA: podíl mobilní aplikace v letech 2025 a 2026

# =====================================================================================
# 11) CVIČENÍ
# =====================================================================================
# ---- Cvičení 1: sezónní heatmapa – kategorie × měsíc ----
tab_km=(radky[(radky["rok"]==2025)&(radky["kategorie"]!="Vyřazené zboží")]   # jen rok 2025 a bez vyřazeného zboží
        .pivot_table(index="kategorie", columns="měsíc", values="trzba_bez_dph", aggfunc="sum"))   # řádky = kategorie, sloupce = měsíce, hodnota = součet tržeb
index_km=tab_km.div(tab_km.mean(axis=1), axis=0)             # každý řádek vydělíme jeho průměrem (mean(axis=1) = průměr přes měsíce) = sezónní index
fig, ax=plt.subplots(figsize=(11,3.8))                       # plátno
sns.heatmap(index_km, cmap="RdBu_r", center=1, vmin=0, vmax=2,   # divergující škála se středem v 1 (průměrný měsíc)
            annot=True, fmt=".1f", annot_kws={"size":8}, linewidths=0.5,   # čísla v buňkách
            cbar_kws={"label":"Sezónní index (1 = průměrný měsíc)"}, ax=ax)
ax.grid(False)                                               # vypnout mřížku
ax.set_xlabel("Měsíc v roce 2025"); ax.set_ylabel("")        # popisky os
ax.set_title("Únor je nejslabší měsíc téměř všude; Hračky mají v prosinci 3× průměr", loc="left")   # závěr
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_22_cviceni1_sezonni_heatmapa.png", dpi=120)   # uložení
plt.show()                                                   # zobrazení

# ---- Cvičení 2a: doba doručení podle dopravce ----
fig, ax=plt.subplots(figsize=(8,4.5))                        # plátno
sns.boxplot(data=dok, x="dopravce", y="delivery_days", order=["DPD","PPL","Zasilkovna","Ceska posta"],   # boxplot doby doručení pro každého dopravce, v zadaném pořadí
            color=SVETLA_MODRA, fliersize=2, ax=ax)
ax.set_xlabel(""); ax.set_ylabel("Doba doručení (dny)")      # popisky os
ax.set_title("Česká pošta je nejpomalejší; DPD a PPL doručí obvykle za 1–3 dny", loc="left")   # závěr
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_23_cviceni2a_doruceni_dopravce.png", dpi=120)   # uložení
plt.show()                                                   # zobrazení

# ---- Cvičení 2b: DPD po měsících – plošný problém, nebo pár zpožděných zásilek? ----
dpd=dok[dok["dopravce"]=="DPD"].copy()                       # jen objednávky doručované DPD
dpd["rok_mesic_text"]=dpd["rok_mesic"].astype(str)           # období převedeme na text (seaborn ho pak použije jako kategorii na ose x)
fig, ax=plt.subplots(figsize=(13,4.5))                       # široké plátno kvůli 20+ měsícům
sns.boxplot(data=dpd, x="rok_mesic_text", y="delivery_days", color=SVETLA_MODRA, fliersize=2, ax=ax)   # jeden box za každý měsíc
ax.tick_params(axis="x", rotation=60)                        # popisky osy x natočíme o 60°, aby se nepřekrývaly
ax.set_xlabel(""); ax.set_ylabel("Doba doručení (dny) – DPD")   # popisky os
ax.set_title("DPD v prosinci 2025: celý box je posunutý nahoru, nejde o pár zpožděných zásilek, ale o plošný problém", loc="left")   # závěr: posun celého boxu = problém se týká většiny zásilek
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_24_cviceni2b_dpd_mesice.png", dpi=120)   # uložení
plt.show()                                                   # zobrazení

# ---- Cvičení 3: stojí za to víc příznaků? Decily sady A (jen recency) × sady C (všechno) ----
fig, osy=plt.subplots(1, 2, figsize=(12,4.5), sharey=True)   # sharey=True = oba grafy mají stejnou osu y, aby šly poctivě porovnat
for ax, (nazev, sloupce_sady) in zip(osy, [("A: jen recency", ["recency_dny"]), ("C: všechny příznaky", list(X_train_s.columns))]):   # dvě sady: (název, sloupce)
    m=LogisticRegression(max_iter=1000)                      # nový model
    m.fit(X_train_s[sloupce_sady], y_train)                  # učení na vybraných sloupcích
    p=m.predict_proba(X_test_s[sloupce_sady])[:,1]           # pravděpodobnosti na testu
    t=pd.DataFrame({"skutecnost":y_test.values, "p":p})      # tabulka skutečnost × pravděpodobnost
    t["decil"]=pd.qcut(t["p"].rank(method="first"), 10, labels=False)   # 10 stejně velkých skupin podle pravděpodobnosti
    d=t.groupby("decil")["skutecnost"].mean()                # skutečný podíl kupců v každém decilu
    ax.bar(d.index+1, d.values, color=MODRA, width=0.7)      # sloupce 1–10
    ax.axhline(y_test.mean(), color=SEDA, linestyle="--", linewidth=1)   # celkový průměr
    ax.set_xticks(range(1,11)); ax.set_xlabel("Decil (1 = nejnižší pravděpodobnost)")   # osa x
    ax.set_title(f"Sada {nazev}: rozdíl nejhorší × nejlepší decil = {d.iloc[-1]-d.iloc[0]:.0%} b.", loc="left")   # rozdíl mezi nejlepším a nejhorším decilem v procentních bodech (b.)
osy[0].set_ylabel("Podíl, kdo koupil znovu"); osy[0].set_ylim(0,1.05)   # popisek a rozsah osy y (sdílená)
fig.tight_layout()                                           # srovnání mezer
fig.savefig("outputs/v2_25_cviceni3_decily_A_C.png", dpi=120)   # uložení
plt.show()                                                   # zobrazení
