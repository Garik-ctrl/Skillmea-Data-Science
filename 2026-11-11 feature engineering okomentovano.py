# =====================================================================================
# LEKCE 3 – FEATURE ENGINEERING (hodina z 9. 11. 2026) – OKOMENTOVÁNO
# =====================================================================================
#
import pandas as pd                                                  # pandas = tabulky (DataFrame); běžně se importuje jako pd
from matplotlib import pyplot as plt                                 # pyplot = kreslení grafů; (znovu se importuje níže jako plt – nevadí, jen se opakuje)

# -------------------------------------------------------------------------------------
# 1) NAČTENÍ DAT
# -------------------------------------------------------------------------------------
# r"..." = "raw string": zpětná lomítka v cestě k souboru se nebrána jako speciální znaky (\n, \t ...)
objednavky=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\objednavky2.csv")   # řádky objednávek (1 řádek = 1 položka)
produkty=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\produkty2.csv")       # katalog produktů (kategorie, cena)
zakaznici=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\zakaznici2.csv")     # zákazníci (kraj, segment, registrace ...)

print(objednavky.head())                                             # .head() = prvních 5 řádků, ať vidíme, jak data vypadají
print(zakaznici.head())

print(objednavky.info())                                             # .info() = sloupce, datové typy, počet neprázdných hodnot, paměť
print(zakaznici.info())

# -------------------------------------------------------------------------------------
# 2) DATUMY: text -> skutečné datum
# -------------------------------------------------------------------------------------
# V CSV je datum jen text. Až po převodu umíme odečítat data, brát .dt.days, .dt.month, porovnávat s hranicí.
objednavky["order_date"]=pd.to_datetime(objednavky["order_date"], format="%Y-%m-%d %H:%M")   # format říká pandasu přesně, jak text vypadá (rok-měsíc-den hodina:minuta) -> rychlejší a bez hádání
zakaznici["registrace"]=pd.to_datetime(zakaznici["registrace"], format="%Y-%m-%d")           # tady je jen datum bez času

# -------------------------------------------------------------------------------------
# 3) SPOJENÍ TABULEK (merge = v Excelu VLOOKUP / v SQL JOIN)
# -------------------------------------------------------------------------------------
#objednávky + produkty
op=objednavky.merge(produkty,on="product_id", how="left", validate="m:1")
# on="product_id"  = podle jakého sloupce se tabulky párují
# how="left"       = zachovej VŠECHNY řádky objednávek (i kdyby produkt v katalogu chyběl -> doplní se prázdné hodnoty)
# validate="m:1"   = pojistka: "many to one" – mnoho objednávek ku jednomu produktu. Kdyby měl katalog produkt dvakrát,
#                    pandas spadne s chybou místo toho, aby tiše zdvojil řádky.
assert len(op)==len(objednavky)                                      # KONTROLA: po spojení nesmí přibýt ani ubýt řádek (assert = "tohle musí platit, jinak zastav")
print(op.info())
print(op.head())

#pd.set_option("display.max_columns",40)
#pd.set_option("display.width",2000)
#objednávky + produkty + zákazníci
opz=op.merge(zakaznici,on="customer_id", how="left", validate="m:1")   # ke každé položce přidáme údaje o zákazníkovi (m:1 = zákazník může mít víc položek)
assert len(opz)==len(op)                                             # KONTROLA: opět se nezměnil počet řádků
print(opz.head())

# -------------------------------------------------------------------------------------
# 4) TRŽBA ŘÁDKU
# -------------------------------------------------------------------------------------

op["trzba_bez_dph"]=op["quantity"]*op["cena_za_kus_bez_dph"]*(1-op["discount_pct"]/100)
# kusy × cena za kus × (1 – sleva v procentech/100). Sleva je v datech jako 12 (= 12 %), proto /100.
# Počítá se po sloupcích najednou (vektorově) – žádný cyklus není potřeba.

# -------------------------------------------------------------------------------------
# 5) TABULKA OBJEDNÁVEK (řádky -> 1 řádek = 1 objednávka)
# -------------------------------------------------------------------------------------

agg_tabulka=(op                                                      # závorky ( ) kolem celého řetězce umožní psát každou metodu na nový řádek
             .groupby("order_id")                                    # rozděl řádky do skupin podle čísla objednávky
             .agg(order_date=("order_date", "first"),                # pojmenovaná agregace: nový_sloupec=("zdrojový_sloupec","funkce"); "first" = hodnota z prvního řádku skupiny
                  customer_id=("customer_id", "first"),              # údaje o objednávce jsou na všech jejích řádcích stejné -> stačí první
                  status=("status", "first"),
                  channel=("channel", "first"),
                  payment=("payment", "first"),
                  dopravce=("dopravce", "first"),
                  delivery_days=("delivery_days", "first"),
                  pocet_polozek=("radek", "count"),                  # "count" = kolik řádků (položek) objednávka má
                  kusy=("quantity", "sum"),                          # "sum" = součet kusů za všechny položky
                  trzba_bez_dph=("trzba_bez_dph", "sum"),            # tržba objednávky = součet tržeb jejích položek
                  sleva_prumer=("discount_pct", "mean"))             # "mean" = aritmetický průměr slevy přes položky (neváženě – nevadí, jen orientační)
             .reset_index()                                          # order_id je po groupby v indexu -> vrátíme ho zpět jako obyčejný sloupec
             .reset_index(drop=True)                                 # přečísluje řádky 0,1,2...; (po prvním reset_index je to zbytečné, nic nemění)
             .sort_values(["order_date","order_id"])                 # seřadit podle data a při shodě podle čísla objednávky
)
print(agg_tabulka)

# -------------------------------------------------------------------------------------
# 6) HRANICE: HISTORIE × BUDOUCNOST  (PRAVIDLO 1)
# -------------------------------------------------------------------------------------
hranice=pd.Timestamp("2026-06-30 23:59")                             # Timestamp = jedno konkrétní datum a čas

zacatek=pd.Timestamp("2026-06-30 23:59")
konec=pd.Timestamp("2026-10-31 23:59")                               # konec dat
rozdil=konec-zacatek                                                 # rozdíl dvou dat = Timedelta (časový úsek)
print(rozdil.days)                                                   # KONTROLA: 123 dnů = délka období, ve kterém sledujeme, jestli zákazník koupí znovu
testovaci_obdobi=123                                                 # to samé číslo "natvrdo"; proměnná se dál nepoužívá

import numpy as np                                                   # numpy = matematika nad poli; np.where, np.log1p ...
agg_tabulka["obdobi"]=np.where(agg_tabulka["order_date"]<=hranice, "historicka data", "budoucnost")
# np.where(podmínka, hodnota_když_platí, hodnota_když_neplatí) = "IF" pro celý sloupec najednou
print(agg_tabulka["obdobi"].value_counts())                          # .value_counts() = kolik řádků má každá hodnota; ověříme poměr historie/budoucnost

print(agg_tabulka)

# -------------------------------------------------------------------------------------
# 7) FILTRY: koho a co do analýzy bereme
# -------------------------------------------------------------------------------------
registrovani=agg_tabulka.dropna(subset=["customer_id"]).copy()      # zahodí řádky, kde chybí customer_id (hosté bez účtu – nejdou sledovat v čase). V čisté verzi hosté nejsou, ale zvyk je správný.
                                                                     # .copy() = samostatná kopie; vyhneme se varování SettingWithCopyWarning při dalších úpravách
dokoncene_obchody=registrovani[registrovani["status"]=="dokonceno"].copy()   # jen skutečně dokončené objednávky (ne zrušené / vrácené)
print(dokoncene_obchody)
obchody_historie=dokoncene_obchody[dokoncene_obchody["obdobi"]=="historicka data"].copy()   # dokončené objednávky před hranicí -> ZDROJ PŘÍZNAKŮ
obchody_budoucnost=dokoncene_obchody[dokoncene_obchody["obdobi"]=="budoucnost"].copy()      # dokončené objednávky po hranici -> ZDROJ CÍLE

# -------------------------------------------------------------------------------------
# 8) POPULACE ZÁKAZNÍKŮ A CÍL (y)
# -------------------------------------------------------------------------------------
zakaznik=(pd.DataFrame({"customer_id":obchody_historie["customer_id"].unique()})   # .unique() = každé customer_id jednou; z toho uděláme tabulku s 1 sloupcem
          .set_index("customer_id")                                  # customer_id se stane indexem (štítkem řádků) -> později snadné .join
          .sort_index())                                             # seřadit podle indexu (stabilní pořadí)
print(zakaznik)

zakaznik["koupi_znovu"]=zakaznik.index.isin(obchody_budoucnost["customer_id"].unique()).astype(int)
# .isin(seznam) = True, pokud je zákazník v seznamu těch, kdo nakoupili v budoucnosti
# .astype(int)  = True/False -> 1/0 (to je náš CÍL y: 1 = koupí znovu, 0 = nekoupí)
print("Počet zákazníků ", len(zakaznik), ". Podíl zákazníků, kteří koupí znovu ", round(zakaznik["koupi_znovu"].mean(),2 ))
# .mean() ze sloupce samých 0 a 1 = podíl jedniček; round(...,2) = na 2 desetinná místa

# -------------------------------------------------------------------------------------
# 9) PŘÍZNAKY RFM (Recency – Frequency – Monetary) – jen z HISTORIE
# -------------------------------------------------------------------------------------
nakupy=(obchody_historie
        .groupby("customer_id")                                      # skupina = jeden zákazník
        .agg(posledni_nakup=("order_date","max"),                    # "max" z dat = nejpozdější nákup (podklad pro recency)
             frekvence=("order_id","count"),                         # Frequency: kolik dokončených objednávek zákazník do hranice měl
             monetary=("trzba_bez_dph", "sum"))                      # Monetary: kolik celkem utratil (bez DPH)
        )
print(nakupy)
zakaznik=zakaznik.join(nakupy)                                       # .join = spojení podle INDEXU (obě tabulky mají index customer_id)
print(zakaznik)

#průměrná hodnota objednávky
zakaznik["AVG_hodnota_objednavky"]=zakaznik["monetary"]/zakaznik["frekvence"]   # utraceno / počet objednávek = kolik průměrně utratí za jednu objednávku

# -------------------------------------------------------------------------------------
# 10) PRVNÍ MODEL (jen frekvence, monetary, průměrná objednávka)
# -------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split                 # rozdělení dat na učící (train) a testovací (test) část
from sklearn.linear_model import LogisticRegression                  # logistická regrese = model pro předpověď ANO/NE jako pravděpodobnosti
from sklearn.metrics import roc_auc_score                            # AUC = kvalita ŘAZENÍ zákazníků podle pravděpodobnosti

y=zakaznik["koupi_znovu"]  #cíl, který předpovídáme
X=zakaznik.drop(columns=["koupi_znovu","posledni_nakup"])
# X = příznaky. Z tabulky vyhodíme cíl (jinak by model "opsal řešení") a datum posledního nákupu (model nepočítá s daty, jen s čísly;
# z data uděláme později číslo = recency).

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# test_size=0.2 = 20 % zákazníků jde stranou jako "zkouška", 80 % se model učí
# random_state=42 = pevné "zrno" náhody -> při každém spuštění vyjde stejné rozdělení (a obě naše verze modelu se měří na stejných lidech)

median=X_train.median()                                              # medián každého sloupce – POČÍTÁ SE JEN Z TRAINU (pravidlo 2)
X_train = X_train.fillna(median)                                     # prázdné hodnoty doplníme mediánem sloupce
X_test = X_test.fillna(median)                                       # test dostane TENTÝŽ medián z trainu (ne vlastní!)

prumer=X_train.mean()                                                # průměr každého sloupce z trainu
odchylka=X_train.std()                                               # směrodatná odchylka z trainu
X_train_s=(X_train-prumer)/odchylka                                  # standardizace (z-skóre): (hodnota – průměr)/odchylka -> všechny sloupce mají průměr 0 a rozptyl 1
X_test_s=(X_test-prumer)/odchylka                                    # test se škáluje čísly z trainu

print(X_train_s)

model = LogisticRegression()                                         # prázdný model s výchozím nastavením
model.fit(X_train_s,y_train)                                         # .fit = UČENÍ: model hledá váhy příznaků, které nejlépe vysvětlí y_train
accuracy=model.score(X_test_s,y_test)                                # .score = PŘESNOST: podíl zákazníků v testu, které model trefil (0/1)
print(zakaznik)
print(accuracy)
print(model.predict(X_test_s))                                       # .predict = tvrdý tip 0/1 pro každého zákazníka v testu
print(y_test)                                                        # skutečnost, se kterou tip porovnáváme

# -------------------------------------------------------------------------------------
# 11) EXPORT SPOJENÝCH DAT
# -------------------------------------------------------------------------------------
import os
print(opz)
os.makedirs("outputs", exist_ok=True) #vytvoří složku
opz.to_csv("outputs/opz_spojena_data.csv",index=False,encoding="utf-8-sig")

# -------------------------------------------------------------------------------------
# 12) NOVÉ PŘÍZNAKY: RECENCY A INTERVAL MEZI NÁKUPY
# -------------------------------------------------------------------------------------
#počet dní od posledního nákupu
zakaznik["Dny_od_posledniho_nakupu"]=(hranice-zakaznik["posledni_nakup"]).dt.days
# hranice – datum = Timedelta (časový úsek); .dt.days = z něj jen celé dny jako číslo
assert zakaznik["Dny_od_posledniho_nakupu"].min()>=0                 # KONTROLA: žádný záporný údaj -> nikdo nenakoupil PO hranici (pravidlo 1 drží)
print(zakaznik.head())

serazene_obchody=obchody_historie.sort_values(by=["customer_id", "order_date"]).copy()   # seřadit: nejdřív podle zákazníka, uvnitř podle času (diff potřebuje správné pořadí)
serazene_obchody["interval"]=serazene_obchody.groupby("customer_id")["order_date"].diff().dt.days
# .groupby("customer_id")[...].diff() = rozdíl data oproti PŘEDCHOZÍ objednávce téhož zákazníka (u první objednávky = NaN, předchůdce nemá)
# .dt.days = rozdíl převedený na celé dny

zakaznik["interval_prumer"]=serazene_obchody.groupby("customer_id")["interval"].mean()
# průměrný odstup mezi nákupy; .mean() prázdné hodnoty (NaN) přeskakuje
# Přiřazení funguje správně díky tomu, že oba objekty mají index customer_id (pandas přiřazuje podle indexu).

print(zakaznik["interval_prumer"].isna().sum())                      # .isna() = True kde je prázdno; .sum() = kolik jich je
# OTÁZKA: Dá se místo mediánu použít 0? (ne – 0 by znamenalo "nakupuje každý den"; prázdné je jiná informace než nula)

# -------------------------------------------------------------------------------------
# 13) DRUHÝ MODEL – s novými příznaky + měření pomocí AUC
# -------------------------------------------------------------------------------------
# POZOR: po přidání příznaků musíme X, split, medián i škálování udělat znovu – jinak nové sloupce v X_train nejsou.
y=zakaznik["koupi_znovu"]  #cíl, který předpovídáme
X=zakaznik.drop(columns=["koupi_znovu","posledni_nakup"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # stejné random_state=42 -> stejní zákazníci v testu -> modely jsou férově srovnatelné

median=X_train.median()
X_train = X_train.fillna(median)                                     # interval_prumer se doplní mediánem
X_test = X_test.fillna(median)

prumer=X_train.mean()
odchylka=X_train.std()
X_train_s=(X_train-prumer)/odchylka
X_test_s=(X_test-prumer)/odchylka

print(X_train_s)

model = LogisticRegression()
model.fit(X_train_s,y_train)
accuracy=model.score(X_test_s,y_test)
print(accuracy)
AUC=roc_auc_score(y_test, model.predict_proba(X_test_s)[:,1])
# model.predict_proba(X) = pravděpodobnosti pro obě třídy (sloupec 0 = nekoupí, sloupec 1 = koupí); [:,1] = vezmi jen sloupec "koupí"
# roc_auc_score(skutečnost, skóre) = AUC: "Vezmu náhodného kupce a náhodného nekupce. Jak často dá model kupci vyšší skóre?"
#                                    0,5 = házení mincí, 1,0 = dokonalé
AUC_accuracy=roc_auc_score(y_test,-X_test["Dny_od_posledniho_nakupu"])
# Srovnávací "model" bez učení: skóre = jen recency. Minus před sloupcem, protože MENŠÍ recency = VĚTŠÍ šance na nákup.
print(AUC, AUC_accuracy)

# -------------------------------------------------------------------------------------
# 14) SKÓRE PRO VŠECHNY ZÁKAZNÍKY A SEGMENTACE
# -------------------------------------------------------------------------------------
#skore pro všechny zákazníky

X_vse=X.fillna(median)                                               # všichni zákazníci (train i test); chybějící hodnoty doplněny mediánem z trainu
print(X_vse)
zakaznik["pravdepodobnost"]= model.predict_proba((X_vse-prumer)/odchylka)[:,1]   # stejné škálování jako u trainu -> pravděpodobnost, že zákazník koupí znovu
zakaznik["riziko"]=pd.cut(zakaznik["pravdepodobnost"]
                          ,bins=[0,0.4,0.7,1]
                          , labels=["Zrádci","Nevyhranění", "Věrní"]) # dělení na "chlívečky" - vytvoříme skupiny s pevnými hranicemi
# pd.cut(sloupec, bins=hranice, labels=názvy): rozdělí spojité číslo do skupin: (0 – 0,4] = Zrádci, (0,4 – 0,7] = Nevyhranění, (0,7 – 1] = Věrní.
# Intervaly jsou zprava uzavřené. Počet názvů musí být o 1 menší než počet hranic.
zakaznik.to_csv("outputs/zakaznik_skore.csv",index=False,encoding="utf-8")

# -------------------------------------------------------------------------------------
# 15) VIZUALIZACE – začátek (pokračuje příští hodinu: Vizualizace dat I)
# -------------------------------------------------------------------------------------
#Graf č.1 Tržba po měsících
print(dokoncene_obchody.info())
mesicni_trzby=round(dokoncene_obchody.groupby(dokoncene_obchody["order_date"].dt.to_period("M"))["trzba_bez_dph"].sum()/1e6,2) #1e6 - podělit milionem
# Řetězec čtený zevnitř ven:
#   dokoncene_obchody["order_date"].dt.to_period("M")  = z data uděláme "rok-měsíc" (např. 2026-03) – skupina pro každý měsíc
#   .groupby(...)["trzba_bez_dph"].sum()               = součet tržeb objednávek v každém měsíci
#   /1e6                                               = 1e6 = 1 000 000 -> hodnoty v milionech Kč
#   round(...,2)                                       = na 2 desetinná místa
print(mesicni_trzby.info())

#dokoncene_obchody["rok-mesic"]=dokoncene_obchody["order_date"].dt.to_period("M")
#mesicni_trzby=dokoncene_obchody.groupby(dokoncene_obchody["rok-mesic"])["trzba_bez_dph"].sum()
import matplotlib.ticker as mticker                                  # mticker = formátování popisků na osách

fig, ax = plt.subplots(figsize=(9,4))
# plt.subplots() vytvoří prázdný obrázek (fig = celé plátno) a jednu kreslicí plochu (ax = osy, na které kreslíme); figsize = šířka a výška v palcích
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f} mil. Kč"))   # popisky osy y: číslo s tisícovým oddělovačem, 2 desetinná místa a jednotkou
ax.plot(mesicni_trzby.index.to_timestamp(),                          # osa x: období (2026-03) převedené na datum (1. den měsíce), protože osa x potřebuje data
        mesicni_trzby.values,                                        # osa y: hodnoty tržeb (.values = holé číslo bez indexu)
        color="blue",                                                # barva čáry
        marker="o",                                                  # kolečko v každém bodě
        markersize=5,                                                # velikost kolečka
        label="trzba bez_dph",                                       # název řady pro legendu
        linewidth=2)                                                 # tloušťka čáry
ax.axvline(hranice,linestyle="--", color="grey", linewidth=1)        # axvline = svislá čára v daném místě osy x (hranice historie/budoucnost); "--" = čárkovaná
ax.text(hranice,mesicni_trzby.max(),"hranice", va="top")             # text na pozici (x, y); va="top" = horní okraj textu je na dané výšce
ax.set_ylabel("trzby bez_dph")                                       # popisek osy y
ax.set_title("Trzby bez_dph")                                        # titulek grafu
fig.savefig("outputs/mesicni_trzby.png")
print(plt.show())

# sloupcový graf - tržby podle kanálu
print(dokoncene_obchody.info())
Korporatni_barva_c1="#A17B74"                                        # barva zapsaná šestnáctkově (HEX)

trzby_dle_kanalu=round(dokoncene_obchody.groupby("channel")["trzba_bez_dph"].sum().sort_values()/1e6,2)
# groupby("channel") = skupina pro každý kanál (web, telefon, mobilní aplikace ...); .sum() = tržba kanálu;
# .sort_values() = seřadit vzestupně (u vodorovných sloupců se první kreslí dole -> největší bude nahoře); /1e6 = miliony; round = 2 místa
print(trzby_dle_kanalu)
fig, ax = plt.subplots(figsize=(9,4))
ax.barh(trzby_dle_kanalu.index,                                      # barh = vodorovné sloupce (h = horizontal); názvy kanálů na ose y
       trzby_dle_kanalu.values, color= "#A17B74")                    # délka sloupce = tržba
fig.savefig("outputs/trzby_dle_kanalu.png")                          # uložení do souboru