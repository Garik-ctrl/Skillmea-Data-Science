# Co tento soubor dělá, ve čtyřech krocích:
#   1. načte tři tabulky (objednávky, produkty, zákazníci), spojí je a udělá tabulku objednávek
#   2. určí HRANICI ("dnes"), rozdělí objednávky na historii a budoucnost a vyrobí CÍL (koupi_znovu)
#   3. vyrobí první příznaky (RFM) – tabulka, kde 1 řádek = 1 zákazník
#   4. rozdělí na train/test, doplní/škáluje podle trainu a natrénuje první model

# ---------- 1) NAČTENÍ DAT ----------
import pandas as pd
# read_csv načte CSV do tabulky; r"..." (raw string) říká, že zpětná lomítka v cestě nejsou speciální znaky
objednavky=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\objednavky2.csv")   # 1 řádek = 1 POLOŽKA objednávky
produkty=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\produkty2.csv")       # katalog produktů (kategorie, cena, náklad)
zakaznici=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\zakaznici2.csv")     # zákazníci (kraj, segment, rok narození, registrace)


print(objednavky.head())
print(zakaznici.head())

print(objednavky.info())
print(zakaznici.info())

# to_datetime převede text na datum; format říká PŘESNĚ, jak text vypadá (rok-měsíc-den hodina:minuta)
objednavky["order_date"]=pd.to_datetime(objednavky["order_date"], format="%Y-%m-%d %H:%M")   # bez převodu bychom datumy nešlo odečítat
zakaznici["registrace"]=pd.to_datetime(zakaznici["registrace"], format="%Y-%m-%d")           # totéž pro registraci (tady bez hodin)

# ---------- 1.1 SPOJENÍ TABULEK ----------
#objednávky + produkty
# merge = spojení podle klíče; how="left" = zachovat všechny řádky objednávek; validate="m:1" = v produktech smí být každý klíč jen jednou
op=objednavky.merge(produkty,on="product_id", how="left", validate="m:1")
assert len(op)==len(objednavky)                              # KONTROLA po každém merge: počet řádků se nesmí změnit
print(op.info())                                             # KONTROLA: přibyly sloupce z produktů (kategorie, cena_bez_dph, ...)
print(op.head())                                             # pohled na spojenou tabulku

#pd.set_option("display.max_columns",40)                     # (zakomentováno) kdyby se ve výpisu zkracoval počet sloupců, odkomentuj
#pd.set_option("display.width",2000)                         # (zakomentováno) šířka výpisu v konzoli
#objednávky + produkty + zákazníci
opz=op.merge(zakaznici,on="customer_id", how="left", validate="m:1")   # přidá kraj, segment, rok narození, registraci
assert len(opz)==len(op)                                     # KONTROLA: ani druhý merge nezměnil počet řádků
print(opz.head())                                            # pohled na kompletní tabulku
# POZOR: dál pracujeme s tabulkou op (BEZ zákazníků), ne s opz. Informace o zákaznících si později připojíme
#        samostatně podle customer_id (např. věk a segment). opz teď slouží jen jako kontrola, že merge funguje.

# ---------- 1.2 TRŽBA A TABULKA OBJEDNÁVEK ----------
#trzba_bez_dph
# tržba jednoho řádku = kusy × cena za kus × (1 − sleva v %/100); počítá se po celých sloupcích najednou
op["trzba_bez_dph"]=op["quantity"]*op["cena_za_kus_bez_dph"]*(1-op["discount_pct"]/100)
print(op.head())

# Z řádků (položek) na objednávky: 1 řádek = 1 objednávka.
# groupby("order_id") = jedna skupina na objednávku; agg(nový_sloupec=("zdrojový_sloupec", "funkce"))
#   "first" = hodnota z prvního řádku objednávky (datum, zákazník, kanál jsou na všech řádcích objednávky stejné)
#   "count" = počet řádků, "sum" = součet, "mean" = průměr
agg_tabulka=(op                                              # závorka dovolí napsat řetězec metod na více řádků
             .groupby("order_id")                            # rozdělí řádky do skupin podle čísla objednávky
             .agg(order_date=("order_date", "first"),        # datum objednávky
                  customer_id=("customer_id", "first"),      # zákazník (u hosta by bylo prázdné; v čisté verzi hosté nejsou)
                  status=("status", "first"),                # dokonceno / vraceno / zruseno
                  channel=("channel", "first"),              # web / mobilni_app / telefon
                  payment=("payment", "first"),              # způsob platby
                  dopravce=("dopravce", "first"),            # dopravce
                  delivery_days=("delivery_days", "first"),  # doba doručení ve dnech (u zrušených prázdná – nemají doručení)
                  pocet_polozek=("radek", "count"),          # počet položek v objednávce
                  kusy=("quantity", "sum"),                  # součet kusů
                  trzba_bez_dph=("trzba_bez_dph", "sum"),    # tržba objednávky = součet tržeb položek
                  sleva_prumer=("discount_pct", "mean"))     # průměrná sleva přes položky
             .reset_index()                                  # order_id z indexu zpět do sloupce
             .reset_index(drop=True)                         # (druhý reset_index je tu navíc – nic nemění, můžeš smazat)
             .sort_values(["order_date","order_id"])         # seřadit podle času; potom už řádky nejsou číslované popořadě
)
print(agg_tabulka)                                           # KONTROLA: čistý Vlnka Shop má 26 103 objednávek

# ---------- 2) HRANICE, HISTORIE A BUDOUCNOST ----------
# LEKTOR: Nakresli na tabuli osu času – je to nejdůležitější obrázek lekce:
#   2025-01-01 ............... HRANICE (30. 6. 2026) ............... 2026-10-31
#   |<------ HISTORIE ------>|<------- BUDOUCNOST (4 měsíce) ------->|
#   |  z toho PŘÍZNAKY (X)   |  z toho CÍL (y)                       |
hranice=pd.Timestamp("2026-06-30 23:59")                     # "dnes" z pohledu modelu; Timestamp = jedno konkrétní datum a čas

zacatek=pd.Timestamp("2026-06-30 23:59")                     # začátek testovacího okna (shodný s hranicí – tato proměnná je duplicitní)
konec=pd.Timestamp("2026-10-31 23:59")                       # konec dat = konec okna cíle
rozdil=konec-zacatek                                         # odečtení dvou dat = Timedelta (časový rozdíl)
print(rozdil.days)                                           # .days = celé dny; KONTROLA: 123
testovaci_obdobi=123                                         # délka okna cíle ve dnech (zatím se dál nepoužívá)
# TIP: konec okna šlo spočítat jako hranice+pd.Timedelta(days=testovaci_obdobi) a použít ho při filtrování budoucnosti
#      (viz POZOR níže); proměnná konec by pak nebyla třeba.

import numpy as np                                           # numpy = matematika nad celými sloupci (where, select, log, ...)
# np.where(podmínka, když_ano, když_ne) = IF pro celý sloupec; objednávka do hranice = historie, později = budoucnost
agg_tabulka["obdobi"]=np.where(agg_tabulka["order_date"]<=hranice, "historicka data", "budoucnost")
print(agg_tabulka["obdobi"].value_counts())                  # value_counts() = kolik řádků je v každé skupině

print(agg_tabulka)                                           # KONTROLA: nový sloupec obdobi

# ---------- 2.1 POPULACE A CÍL ----------
registrovani=agg_tabulka.dropna(subset=["customer_id"]).copy()   # dropna zahodí objednávky bez zákazníka (hosté); copy = vlastní kopie
dokoncene_obchody=registrovani[registrovani["status"]=="dokonceno"].copy()   # jen dokončené (vrácené a zrušené se nepočítají jako nákup)
print(dokoncene_obchody)                                     # KONTROLA: ve sloupci status jsou jen "dokonceno"
obchody_historie=dokoncene_obchody[dokoncene_obchody["obdobi"]=="historicka data"].copy()   # z těchto objednávek budou PŘÍZNAKY
obchody_budoucnost=dokoncene_obchody[dokoncene_obchody["obdobi"]=="budoucnost"].copy()      # z těchto objednávek bude CÍL
# Upozornění:
#   budoucnost tu nemá horní mez (order_date <= konec okna). U tohoto datasetu to nevadí, protože data končí 31. 10.,
#   ale u jiných dat by cíl závisel na tom, kde data náhodou končí.


# Populace = zákazníci, kteří v historii aspoň jednou nakoupili (bez nákupu není "znovu")
zakaznik=(pd.DataFrame({"customer_id":obchody_historie["customer_id"].unique()})   # unique() = každý zákazník jednou
          .set_index("customer_id")                          # ID zákazníka se stane indexem (jménem řádku)
          .sort_index())                                     # seřadit podle ID, ať je pořadí stabilní
print(zakaznik)                                              # KONTROLA: 2 742 zákazníků, zatím bez sloupců

# Cíl: koupil zákazník v okně po hranici znovu?  isin(seznam) = True/False, zda je ID v seznamu; astype(int) = 0/1
zakaznik["koupi_znovu"]=zakaznik.index.isin(obchody_budoucnost["customer_id"].unique()).astype(int)
# mean() z 0/1 sloupce = podíl jedniček = podíl zákazníků, kteří koupí znovu
print("Počet zákazníků ", len(zakaznik), ". Podíl zákazníků, kteří koupí znovu ", round(zakaznik["koupi_znovu"].mean(),2 ))   # KONTROLA: 2742, 0,62
# OTÁZKA: Co kdyby bylo 99 % jedniček? (pak by "vždy tipuj 1" mělo 99 % a nic bychom se nenaučili.)
#         Tady je to 62 %: tohle číslo je naše "základní" přesnost, kterou musí model překonat (viz konec souboru).

# ---------- 3) PŘÍZNAKY – PRVNÍ DÁVKA: RFM ----------
nakupy=(obchody_historie                                     # vzniká jen z HISTORIE (pravidlo 1)
        .groupby("customer_id")                              # skupina = jeden zákazník
        .agg(posledni_nakup=("order_date","max"),            # max u data = nejnovější nákup (z toho bude Recency)
             frekvence=("order_id","count"),                 # Frequency: počet objednávek
             monetary=("trzba_bez_dph", "sum"))              # Monetary: kolik celkem utratil
        )
print(nakupy)                                                # KONTROLA: 1 řádek = 1 zákazník
zakaznik=zakaznik.join(nakupy)                               # join = připojí podle indexu (customer_id)
print(zakaznik)                                              # KONTROLA: zakaznik má sloupce koupi_znovu, posledni_nakup, frekvence, monetary

#průměrná hodnota objednávky
zakaznik["AVG_hodnota_objednavky"]=zakaznik["monetary"]/zakaznik["frekvence"]   # AOV = tržba / počet objednávek
# TIP: z posledni_nakup se má stát RECENCY: (hranice - zakaznik["posledni_nakup"]).dt.days
#      V kódu níže se sloupec posledni_nakup jen zahodí – a tím přijdeme o NEJSILNĚJŠÍ příznak. Přidá ho pokračování.

# ---------- 4) TRAIN / TEST A PRVNÍ MODEL ----------
from sklearn.model_selection import train_test_split         # funkce na rozdělení dat na trénovací a testovací část
from sklearn.linear_model import LogisticRegression          # jednoduchý model: logistická regrese (odpověď ano/ne)
from sklearn.metrics import roc_auc_score                    # AUC – zatím naimportováno, ale nepoužito (viz pokračování)

y=zakaznik["koupi_znovu"]  #cíl, který předpovídáme
# X = příznaky. posledni_nakup se vyhazuje, protože model potřebuje čísla (ne datum); cíl v X být nesmí!
X=zakaznik.drop(columns=["koupi_znovu","posledni_nakup"])

# test_size=0.2: pětina zákazníků půjde do testu; random_state=42: stejné rozdělení pokaždé (každý dostane totéž)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Doplnění prázdných hodnot MEDIÁNEM Z TRAINU – test dostane STEJNÁ čísla (pravidlo 2)
median=X_train.median()                                      # medián každého sloupce jen z trainu
X_train = X_train.fillna(median)                             # doplnit train
X_test = X_test.fillna(median)                               # doplnit test čísly z trainu
# Tady zatím žádné prázdné hodnoty nejsou (RFM je má vždy). Důležité budou, až přibude interval mezi nákupy a věk.

# Škálování (z-skóre): (hodnota − průměr) / směrodatná odchylka; obojí bereme z trainu
prumer=X_train.mean()                                        # průměr každého sloupce z trainu
odchylka=X_train.std()                                       # směrodatná odchylka každého sloupce z trainu
X_train_s=(X_train-prumer)/odchylka                          # škálovaný train (průměr 0, odchylka 1)
X_test_s=(X_test-prumer)/odchylka                            # škálovaný test – čísla z TRAINU, proto průměr testu vyjde "zhruba" 0 a tak to má být
# Proč škálovat: logistická regrese by jinak "přeceňovala" sloupce s velkými čísly (koruny) oproti malým (kusy).

print(X_train_s)                                             # KONTROLA: hodnoty kolem 0, v rozmezí zhruba −3 až +3

model = LogisticRegression()                                 # prázdný model
model.fit(X_train_s,y_train)                                 # fit = učení; model se učí JEN na trainu
accuracy=model.score(X_test_s,y_test)                        # Upozornění: score() u klasifikace vrací PŘESNOST (accuracy), špatně jsem to pojmenoval
print(zakaznik)                                              # KONTROLA: celá tabulka zákazník × příznaky
print(accuracy)                                               # přesnost: podíl zákazníků v testu, které model uhodl
# OTÁZKA: Je přesnost např. 0,66 dobrá? Odpověď: záleží – "vždy tipuj 1" má přesnost 0,62. Proto potřebujeme AUC (viz pokračování).
print(model.predict(X_test_s))                               # predict = tvrdý odhad 0/1 pro každého zákazníka v testu
print(y_test)                                                # skutečné hodnoty cíle v testu – porovnej s předchozím výpisem