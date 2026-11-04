"""
LEKCE 2. 11. 2026 – Data Analysis I: Čištění dat (cvičení "pronájmy") – okomentovaný přepis

Co jsme se naučili:
  1. vyčistit cenu (text "15 900,- Kč" -> číslo 15900) pomocí regulárního výrazu
  2. vytáhnout z textu dispozici bytu (2+kk, 3+1...) pomocí str.extract
  3. rozdělit lokalitu "Praha - Smíchov" na město a městskou část (str.split)
  4. vytáhnout plochu v m² z textu a převést ji na číslo
  5. převést datum zadané ve dvou různých formátech na jednotný typ datetime
  6. spočítat agregaci po městech (groupby + agg) a vykreslit ji sloupcovým grafem
  7. barevně odlišit sloupce, které stojí na málo inzerátech (n < 2)

Dataset: pronajmy_raw.csv – 14 inzerátů s pronájmem bytů, záměrně "špinavých".
Sloupce: nazev, lokalita, cena, plocha, zverejneno
"""

# ---------------------------------------------------------------
# 0) IMPORTY A NASTAVENÍ VÝPISU
# ---------------------------------------------------------------
import re  # re = modul pro regulární výrazy; tady ho potřebujeme kvůli konstantě re.IGNORECASE
import pandas as pd

pd.set_option("display.max_rows", 20)  # nejvýše 20 řádků ve výpisu (delší tabulky se zkrátí "...")
pd.set_option("display.width", 2000)  # šířka výpisu ve znacích – díky tomu se sloupce nezalamují pod sebe

# ---------------------------------------------------------------
# 1) NAČTENÍ DAT
# ---------------------------------------------------------------
# Seznam textů, které v datech znamenají "hodnota chybí". Pandas je při načtení
# automaticky změní na NaN (= Not a Number, pandas značka pro prázdnou hodnotu).
NuloveHodnoty=['N/A','???','0000', "NULL"]

# read_csv načte soubor do tabulky (DataFrame), kterou ukládáme do proměnné df.
#   r"..."      = raw string: zpětná lomítka ve Windows cestě se nebudou brát jako speciální znaky
#   dtype=str   = VŠECHNY sloupce se načtou jako text. Proč? Aby pandas nic potichu
#                 nepřevedl za nás (např. "05" na číslo 5, nebo datum "5.3.2021" na špatné datum).
#                 Převody pak děláme vědomě, jeden po druhém, a můžeme je kontrolovat.
#   na_values   = náš seznam textů, které se mají změnit na NaN (viz výše)
df= pd.read_csv(r"C:\Users\mistr\Downloads\skillmea_data_analysis_projekt3\pronajmy_raw.csv",
                dtype=str,
                na_values=NuloveHodnoty)

# ---------------------------------------------------------------
# 2) PRVNÍ POHLED NA DATA
# ---------------------------------------------------------------
print(df.head())
print(df.info())
print(df.describe())
print(df.tail())

# ---------------------------------------------------------------
# 3) ČIŠTĚNÍ CENY - 1. MOŽNOST
# ---------------------------------------------------------------
# Zřetězené .str.replace: každý replace odstraní jeden konkrétní text z ceny.
# Princip: df["cena"].str  -> "otevře" textové metody pro celý sloupec
#          .replace(co, čím) -> v každé hodnotě nahradí text "co" textem "čím" ("" = smazat)
# Závorky kolem celého výrazu umožní psát každý krok na nový řádek.
cena_cista=(df["cena"]
            .str.replace(",-","")           # smaže ",-"      "15 900,- Kč" -> "15 900 Kč"
            .str.replace(" Kč","")          # smaže " Kč"     "15 900 Kč"   -> "15 900"
            .str.replace("Kč/měsíc","")     # smaže "Kč/měsíc"
            .str.replace("/měs.",""))       # smaže "/měs."
# Upozornění:
# - záleží na POŘADÍ: u "15 000 Kč/měsíc" už 2. řádek smaže " Kč" dřív,
# než by se 3. řádek mohl trefit do "Kč/měsíc", a zbyde nám "15 000/měsíc".
# - mezery uvnitř čísla ("15 900") zůstávají, takže by to pořád nešlo převést na číslo.

# ---------------------------------------------------------------
# 4) ČIŠTĚNÍ CENY – 2. POKUS (platí)
# ---------------------------------------------------------------
# Místo "co smazat" řekneme "co ponechat": smažeme vše, co NENÍ číslice.
#   r"\D"      = regulární výraz: \D znamená "libovolný znak, který NENÍ číslice" (opak \d)
#   ""         = čím nahradit (ničím = smazat)
#   regex=True = výslovně říkáme, že první parametr je regulární výraz, ne obyčejný text
# Příklady: "15 900,- Kč" -> "15900";  "15900.-" -> "15900";  "cena dohodou" -> "" (prázdný text)
cena_cista = df["cena"].str.replace(r"\D", "", regex=True)
print(cena_cista)
# Upozornění:
# - \D maže VŠECHNO kromě číslic, tedy i desetinnou čárku: "15 900,50" by dalo 1590050.
#U našich dat jsou ceny celé, takže to nevadí – u jiných dat na to pozor.
# - "1 Kč" se promění na 1 a zůstane v datech jako platná cena.
# - Protože 1 Kč je podezdřelá cena, tak ji můžme převést na NaN
#        df.loc[df["cena"] < 1000, "cena"] = pd.NA

# ---------------------------------------------------------------
# 5) DISPOZICE BYTU Z NÁZVU INZERÁTU
# ---------------------------------------------------------------
# str.extract(vzor) vytáhne z každého textu ČÁST odpovídající vzoru (v závorce = zachytávací skupina).
#   (\d+\s*\+\s*(?:kk|\d))  – rozebráno po částech:
#       \d+       jedna nebo více číslic         (např. "2")
#       \s*       nula nebo více mezer           (povolí "2 + kk" i "2+kk")
#       \+        znak plus (v regulárním výrazu má + speciální význam, proto zpětné lomítko)
#       \s*       opět volné mezery
#       (?:kk|\d) buď "kk", nebo jedna číslice; "?:" znamená, že jde o skupinu jen pro výběr
#                 "buď-anebo" a nevrací se zvlášť
#   flags=re.IGNORECASE   velikost písmen nehraje roli ("2+KK" = "2+kk")
#   expand=False          vrátí rovnou jeden sloupec (Series); výchozí expand=True by vrátilo
#                         tabulku (DataFrame) s jedním sloupcem
# Kde se vzor nenajde (např. "Garsonka"), vznikne NaN.
import re
df["dispozice"] = df["nazev"].str.extract(r"(\d+\s*\+\s*(?:kk|\d))", flags=re.IGNORECASE, expand=False)
# \d číslo
# \+ plus
print(df["dispozice"] )
print(df.head())
print(df["lokalita"] )

# ---------------------------------------------------------------
# 6) LOKALITA -> MĚSTO A MĚSTSKÁ ČÁST
# ---------------------------------------------------------------
# Hodnoty vypadají jako "praha - smíchov", "PRAHA 5 -  SMÍCHOV", "brno-Žabovřesky"...
# Řetězec metod se čte shora dolů, každá pracuje s výsledkem předchozí:
mestska_cast=(df["lokalita"]
              .str.title()                  # každé slovo s velkým písmenem: "praha - smíchov" -> "Praha - Smíchov"
                                            # sjednotí velikost písmen, aby "PRAHA" a "praha" byly totéž
              .str.strip()                  # odstraní mezery na začátku a na konci textu
              .str.split(r"\s*-\s*",expand=True))
              # split rozdělí text podle oddělovače. Oddělovač je regulární výraz:
              #   \s*  mezery (i žádné)  +  -  pomlčka  +  \s*  mezery
              #   => "Praha - Smíchov", "Praha-Smíchov" i "Praha -Smíchov" se rozdělí stejně
              # expand=True = výsledek jsou SLOUPCE (0, 1, ...) místo seznamu v jedné buňce
# \s* - mezery
# Upozornění:
# - dělili jsme na hodině jen s krátkou pomlčkou "-". Delší pomlčka "–" (en dash), kterou někdo napsal
#        v inzerátu "Praha 5 – Smíchov", se nerozdělí a do sloupce 0 padne celý text. Totéž
#        způsobuje, že "Praha 5 Smíchov" ×2 se liší jen pomlčkou a nepozná se jako duplicita.
#        Oprava: nejdřív sjednotit pomlčky, pak dělit:
#        .str.replace("–", "-", regex=False)   # vložit PŘED .str.split
# POZOR: kdo má v lokalitě jen město bez části, dostane do sloupce 1 hodnotu NaN.
print(mestska_cast)

print(df.info())
df["mesto"]=mestska_cast[0]          # sloupec 0 = text před pomlčkou
df["mestska_cast"]=mestska_cast[1]   # sloupec 1 = text za pomlčkou
print(df.head())

# ---------------------------------------------------------------
# 7) PLOCHA V m² JAKO ČÍSLO
# ---------------------------------------------------------------
# Plocha je text typu "45 m²", "62m2", "5000 m²". Chceme jen číslo.
# str.extract(r"(\d+)") vezme z textu PRVNÍ souvislou řadu číslic.
#   "45 m²" -> "45";   "62m2" -> "62" (číslice z "m2" už se nebere, bere se jen první řada)
# Výsledek je pořád TEXT ("45"), proto ho následně převádíme na číslo.
df["plocha_m2"]=df["plocha"].str.extract(r"(\d+)")

# to_numeric převede text na číslo. errors="coerce" = co nejde převést, bude NaN
# (místo toho, aby program spadl). Po coerce vždy zkontroluj, kolik NaN přibylo!
df["plocha_m2"]=pd.to_numeric(df["plocha_m2"], errors="coerce")
# Možnost číslo 2
#df["plocha_m2"]=pd.to_numeric(df["plocha"].str.extract(r"(\d+)"), errors="coerce")
print(df.info())

# ---------------------------------------------------------------
# 8) DATUM ZVEŘEJNĚNÍ – DVA FORMÁTY V JEDNOM SLOUPCI
# ---------------------------------------------------------------
# V datech jsou data psaná dvěma způsoby: "05.03.2021" a "2021-03-05".
# Pandas bez udání formátu "hádá", a u data typu 5. 3. 2021 může potichu přečíst
# 3. května. Proto vždy zadáváme formát výslovně:
#   %d = den, %m = měsíc, %Y = rok čtyřciferný; tečky a pomlčky se píší tak, jak jsou v datech
#   errors="coerce" = hodnota, která formátu neodpovídá, se změní na NaT
#                     (NaT = Not a Time, prázdná hodnota pro data)
datum=pd.to_datetime(df["zverejneno"],format="%d.%m.%Y",errors="coerce")
print(datum)   # řádky ve formátu "2021-03-05" jsou teď NaT (nevyhovují formátu s tečkami)

# Druhý pokus s jiným formátem a fillna: tam, kde první pokus dal NaT, doplníme hodnotu
# z druhého pokusu. Tak z obou formátů složíme jeden sloupec.
datum=datum.fillna(pd.to_datetime(df["zverejneno"],format="%Y-%m-%d",errors="coerce"))
# Nemožné datum "31.2.2026" neprojde ani jedním formátem a zůstane NaT – to je správně.
df["zverejneno"]=datum   # přepíšeme původní textový sloupec čistým datem
print(df.info())          # sloupec zverejneno má teď typ datetime64

# ---------------------------------------------------------------
# 9) ODSTRANĚNÍ POMOCNÉHO SLOUPCE A PŘEVOD CENY NA ČÍSLO
# ---------------------------------------------------------------
# drop(columns=[...]) odstraní sloupec. Výsledek vrací jako novou tabulku.
df=df.drop(columns=["plocha"])

# Teď teprve převádíme vyčištěný text z kroku 4 na číslo. Prázdný text ("" z "cena dohodou")
# se přes coerce změní na NaN. Díky tomu "cena dohodou" NENÍ 0 (častá chyba studentů:
# nula by stáhla průměr i medián dolů), ale chybějící hodnota, kterou statistiky ignorují.
df["cena"]=pd.to_numeric(cena_cista, errors="coerce")
print(df["cena"])

print(df.info())  # cena je teď číslo (float64, kvůli NaN) a ne text

# ---------------------------------------------------------------
# 10) AGREGACE PO MĚSTECH
# ---------------------------------------------------------------
#median ceny za m^2 dle města
# groupby("mesto") rozdělí řádky do skupin podle města; agg spočítá pro každou skupinu
# několik ukazatelů najednou. Zápis: nazev_noveho_sloupce=("zdrojovy_sloupec", "funkce")
#   inzeraty             = count nazev -> počet inzerátů ve městě (n)
#   inz_s_cenou          = count cena  -> počet inzerátů, které MAJÍ cenu (count nepočítá NaN)
#   medianova_cena_za_m2 = median cena -> medián ceny (medián = prostřední hodnota po seřazení;
#                          na rozdíl od průměru ho extrémně drahý nebo levný inzerát nepohne)
# sort_values(..., ascending=False) = seřadit od největšího mediánu po nejmenší
median_mesta=(df.groupby("mesto")
              .agg(inzeraty=("nazev", "count"),
                   inz_s_cenou=("cena", "count"),
                   medianova_cena_za_m2=("cena","median"))
              .sort_values("medianova_cena_za_m2", ascending=False))
print(median_mesta)

# Uložení vyčištěné tabulky do CSV.
df.to_csv(r"C:\Users\mistr\Downloads\skillmea_data_analysis_projekt3\pronajmy_raw2.csv"
    , index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------
# 11) GRAF – MEDIÁN PO MĚSTECH, S OZNAČENÍM MÁLO DAT
# ---------------------------------------------------------------
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20,10))

# ax.bar(x, výška, color=...): jeden sloupec na město.
#   x      = názvy měst (index tabulky median_mesta, protože jsme groupovali podle "mesto")
#   výška  = medián z agregace
#   color  = SEZNAM barev, jedna pro každý sloupec. Zde list comprehension:
#            pro každé n (počet inzerátů ve městě) -> "blue" když n>=2, jinak "red".
#            Červený sloupec tedy znamená: "medián stojí jen na jednom inzerátu, nevěř mu".
ax.bar(median_mesta.index
       ,median_mesta["medianova_cena_za_m2"]
       ,color=["blue" if n>=2 else "red" for n in median_mesta["inzeraty"]])

# set_ylim(dole, nahoře): osa Y končí na 130 % nejvyššího sloupce, aby nad ním zbylo místo
ax.set_ylim(0,median_mesta["medianova_cena_za_m2"].max()*1.3)
ax.set_xlabel("Mesta")             # popisek osy X
ax.set_ylabel("Medián nájemného")  # popisek osy Y (po opravě z POZOR 1 by mělo být "Medián Kč/m²")
ax.grid(axis="y", visible=True)    # mřížka jen vodorovná, ať se výšky snáz odečítají
ax.set_title("Median pronájmu v jednotlivých městech")  # nadpis grafu
plt.show()  # zobrazí graf
