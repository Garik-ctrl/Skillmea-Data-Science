"""
LEKCE 4. 11. 2026 – Data Analysis II: Transformace dat – okomentovaný přepis

Co jsme se naučili (v pořadí, jak to v kódu následuje):
  1. načíst 3 propojené CSV (objednávky, produkty, zákazníci) a zjistit, kde chybí hodnoty
  2. převést text na datum (to_datetime s přesným formátem)
  3. pochopit spojování tabulek (merge) na mini příkladu: inner / left / right / outer
  4. vytvořit odvozený sloupec (hrubá tržba = počet kusů * cena)
  5. spojit 3 tabulky do jedné "megatabulky" (LEFT merge) a doplnit chybějící hodnoty (fillna)
  6. vyfiltrovat řádky podle roku (dt.year) a slepit tabulky pod sebe (concat)
  7. spočítat hrubý zisk a cenu s DPH
  8. z order_id "OBJ-000123" udělat číslo (str.replace + astype)
  9. vytáhnout z data rok, měsíc, čtvrtletí, den v týdnu, rozdíl dní (.dt)
 10. filtrovat dokončené objednávky a sečíst je po měsících (groupby)
 11. describe + np.where (rozdělení řádků na "velká" / "běžná")
 12. groupby po objednávkách a po segmentech zákazníků
 13. sloupcový graf počtu zákazníků podle segmentu

Dataset: fiktivní e-shop "Skillmea Shop", 1. 1. 2025 – 31. 10. 2026
  objednavky.csv  ~17 700 řádků = POLOŽKY objednávek (1 objednávka má 1–4 řádky!)
  zakaznici.csv    1 500 řádků  = zákazníci
  produkty.csv        52 řádků  = katalog produktů
"""

# ---------------------------------------------------------------
# 0) IMPORT A NAČTENÍ DAT
# ---------------------------------------------------------------
import pandas as pd  # pandas = knihovna pro práci s tabulkami (DataFrame); "pd" je běžná zkratka

# read_csv načte CSV soubor do tabulky (DataFrame) a uloží ji do proměnné vlevo.
#   r"..." = "raw string": zpětná lomítka ve Windows cestě se nebudou brát jako speciální znaky
#            (bez r by "\U" v "C:\Users" způsobilo chybu)
# Pandas tady odhaduje datové typy sám (na rozdíl od lekce 1, kde jsme použili dtype=str).
# Tip: s relativní cestou by skript fungoval na každém počítači: pd.read_csv("data/objednavky.csv")
objednavky=pd.read_csv(r"C:\Users\mistr\Downloads\skillmea_ekce2\data\objednavky.csv")
produkty=pd.read_csv(r"C:\Users\mistr\Downloads\skillmea_ekce2\data\produkty.csv")
zakaznici=pd.read_csv(r"C:\Users\mistr\Downloads\skillmea_ekce2\data\zakaznici.csv")

# isna() vrátí tabulku True/False (True = buňka je prázdná); sum() pak True sečte (True = 1)
# po sloupcích = kolik prázdných hodnot má každý sloupec.
# Očekávání: customer_id (~3 %, nákup bez registrace), discount_pct (~10 %, sleva nevyplněna),
#            delivery_days (~4 %, zrušené objednávky se nedoručují)
print(objednavky.isna().sum())

# ---------------------------------------------------------------
# 1) AUDIT TABULEK A DATOVÉ TYPY
# ---------------------------------------------------------------
print("objednavky:")
# to_datetime převede TEXT na datum+čas. Díky tomu půjde počítat rozdíly dní, brát rok, měsíc...
#   format říká přesně, jak text vypadá:  %Y = rok čtyřciferný, %m = měsíc, %d = den,
#                                         %H = hodina, %M = minuta (pomlčky, dvojtečka a mezera se píší tak, jak jsou)
#   Pravidlo z lekce 1: formát VŽDY zadávat výslovně, ať pandas nehádá (5. 3. vs. 3. 5.)
objednavky["order_date"]=pd.to_datetime(objednavky["order_date"], format="%Y-%m-%d %H:%M")
print(objednavky["order_date"].head())  # head() = prvních 5 hodnot; kontrola, že vypadají jako datum
print(objednavky.info())  # info() = počet řádků, sloupce, datové typy a počet NEprázdných hodnot
                          # order_date má mít teď typ datetime64[ns] (dřív object = text)
                          # delivery_days je float64, i když jsou to celé dny – kvůli prázdným hodnotám (NaN)


print("zakaznici:")
print(zakaznici.head())
print(zakaznici.info())

zakaznici["registrace"]=pd.to_datetime(zakaznici["registrace"], format="%Y-%m-%d")  # tady je jen datum, bez hodin

print("produkty:")
print(produkty.head())
print(produkty.info())

# ---------------------------------------------------------------
# 2) MINI PŘÍKLAD: JAK FUNGUJE MERGE
# ---------------------------------------------------------------
#mini příklad

# Dvě malé ručně vytvořené tabulky, aby bylo vidět CELÉ, co se při spojení děje.
# DataFrame({"sloupec": [hodnoty], ...}) vyrobí tabulku ze slovníku (klíč = název sloupce).
# obj_mini = objednávky (5 řádků): každá objednávka ví jen KÓD produktu
# kat_mini = katalog (4 řádky): ke kódu produktu zná jeho název
# Produkt P9 je v objednávkách, ale NENÍ v katalogu; produkt P4 je v katalogu,
# ale NIKDO si ho neobjednal. Právě na nich je vidět rozdíl mezi typy merge.
obj_mini=pd.DataFrame({"obj":["A","B","C","D","E"], "prod":["P1","P2","P2", "P3","P9"]})
kat_mini=pd.DataFrame({"prod":["P1","P2","P3","P4"], "nazev":["mys","klavesnice","notebook","monitor"]})
print(obj_mini)
print(kat_mini)

#merge
# merge je v pandasu to, co JOIN v SQL. Parametry:
#   kat_mini   = pravá tabulka (levá je ta, na které merge voláme, tedy obj_mini)
#   on="prod"  = podle kterého sloupce se řádky párují (musí existovat v obou tabulkách)
#   how=...    = které řádky ve výsledku zůstanou:
#       "inner" = jen řádky se shodou V OBOU tabulkách            -> 4 řádky (E/P9 i P4 zmizí)
#       "left"  = všechny řádky z LEVÉ tabulky                    -> 5 řádků (E/P9 zůstane, nazev = NaN)
#       "right" = všechny řádky z PRAVÉ tabulky                   -> 5 řádků (přibude P4, obj = NaN)
#       "outer" = všechno z obou tabulek                          -> 6 řádků (E/P9 i P4)
# Tady je how="outer", takže vidíme obě "sirotčí" hodnoty najednou: E/P9 má prázdný nazev,
# P4/monitor má prázdnou obj. Řádky jsou seřazené podle klíče (P1, P2, P2, P3, P4, P9).
# V praxi se používá z 95 % LEFT (levá tabulka = o čem je analýza, pravá = číselník s popisky).
spojena_mini_tabulka=obj_mini.merge(kat_mini, on="prod", how="outer")
print(spojena_mini_tabulka)
#---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------
# 3) ODVOZENÝ SLOUPEC: HRUBÁ TRŽBA
# ---------------------------------------------------------------
# Nový sloupec = počet kusů * cena za kus. Násobí se celé sloupce najednou (řádek po řádku),
# žádný cyklus for není potřeba (je to řádově rychlejší).
objednavky["hruba_trzba"]=objednavky["quantity"]*objednavky["cena_za_kus_bez_dph"]
print(objednavky.info())  # kontrola: přibyl sloupec hruba_trzba (float64)

# Dvě cesty k témuž číslu: vpravo se tržba spočítá "za běhu" a sečte, dole se sečte hotový sloupec.
# sum() sečte celý sloupec do jednoho čísla. Obě čísla se MUSÍ shodovat – jednoduchá kontrola,
# že jsme sloupec vytvořili správně.
print((objednavky["quantity"]*objednavky["cena_za_kus_bez_dph"]).sum())
print(objednavky["hruba_trzba"].sum())

#---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------
# 4) SPOJENÍ VŠECH TŘÍ TABULEK DO "MEGATABULKY"
# ---------------------------------------------------------------
# len(tabulka) = počet řádků. Ukládáme si ho PŘED merge, abychom ho mohli porovnat PO něm.
pocet_radku_pred=len(objednavky)
print(pocet_radku_pred)

# 1. merge: objednávky + produkty. how="left" = zachovat VŠECHNY řádky objednávek.
# Z katalogu přibudou sloupce (název, kategorie, naklad...). Produkt bez záznamu v katalogu
# dostane do těchto sloupců NaN, ale řádek zůstane (u inner merge by potichu zmizel!).
spojena_tabule_obj_prod=objednavky.merge(produkty, on="product_id", how="left")
print(spojena_tabule_obj_prod.info())  # kontrola: počet řádků stejný jako objednavky, přibyly sloupce z produkty

# 2. merge: výsledek + zákazníci, klíč customer_id.
# Host (customer_id = NaN) nenajde shodu, takže jeho kraj, segment, rok_narozeni a registrace budou NaN.
megatabulka=spojena_tabule_obj_prod.merge(zakaznici, on="customer_id", how="left")
print(megatabulka.info())  # "Non-Null Count" u kraj / segment / registrace je menší než u ostatních = hosté

# fillna(hodnota) doplní prázdné hodnoty zadanou hodnotou.
megatabulka["customer_id"]=megatabulka["customer_id"].fillna("Nezaregistrovan")
megatabulka["rok_narozeni"]=megatabulka["rok_narozeni"].fillna(2100)
megatabulka["rok_narozeni"]=megatabulka["rok_narozeni"].fillna(megatabulka["rok_narozeni"].median())

# ---------------------------------------------------------------
# 5) FILTROVÁNÍ PODLE ROKU A SPOJENÍ POD SEBE (CONCAT)
# ---------------------------------------------------------------

#dt.year - vyextrahuje z toho rok
# .dt zpřístupní části data (jako .str u textu). Funguje jen na typu datetime (proto jsme dělali to_datetime).
# Filtr se skládá ze dvou kroků, čte se zevnitř ven:
#   objednavky["order_date"].dt.year == 2025   -> řada True/False pro každý řádek ("maska")
#   objednavky[ maska ]                        -> ponechá jen řádky, kde je True
rok2025=objednavky[objednavky["order_date"].dt.year==2025]
rok2026=objednavky[objednavky["order_date"].dt.year==2026]

print(objednavky.info())
print(rok2025["order_date"])  # kontrola: ve výpisu jsou jen data z roku 2025
#concat
# concat([a, b]) nalepí tabulky POD SEBE (musí mít stejné sloupce).
#   ignore_index=True = řádky se přečíslují od 0 (jinak by se čísla řádků z obou tabulek opakovala)
# Typické využití: každý měsíc přijde nový soubor se stejnými sloupci, všechny se slepí do jednoho.
spojeno_2025_2026=pd.concat([rok2025,rok2026],ignore_index=True)
# Kontrola: 2025 + 2026 = původní počet řádků (data jsou jen z těchto dvou let; kdyby tam byl i jiný rok, součet by nesedl)
print("pocet zaznamu rok2025 ", len(rok2025), " pocet zaznamu rok2026 ", len(rok2026), " pocet zaznamu dohromady ", len(spojeno_2025_2026))
#---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------
# 6) CHYBĚJÍCÍ SLEVA
# ---------------------------------------------------------------
print(objednavky.isna().sum())  # znovu: kolik prázdných hodnot (nově i sloupce přidané výše)
# Nejdřív sloupec zkopírujeme, ať původní discount_pct zůstane beze změny (se všemi NaN)...
objednavky["chybejici_sleva"]=objednavky["discount_pct"]
# ...a do kopie doplníme chybějící slevu nulou. To je PŘEDPOKLAD (že prázdná sleva = 0 %), ne fakt.
objednavky["chybejici_sleva"]=objednavky["chybejici_sleva"].fillna(0)
print(objednavky["chybejici_sleva"])

#---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------
# 7) HRUBÝ ZISK A CENA S DPH
# ---------------------------------------------------------------
print(megatabulka.info())
print(megatabulka[["quantity","cena_za_kus_bez_dph","naklad"]])  # [[...]] = výběr více sloupců najednou (seznam názvů)
#hrubý zisk = (cena_bez_DPH-naklad)*quantity
# zisk řádku = (prodejní cena - nákladová cena) * počet kusů; počítá se po řádcích najednou.
# naklad pochází z katalogu produktů (merge v oddíle 4). U produktu bez záznamu v katalogu je naklad NaN
# a NaN v součinu dá NaN – správně: zisk neznáme, nevymýšlíme si ho.

megatabulka["hruby_zisk"]=(megatabulka["cena_za_kus_bez_dph"]-megatabulka["naklad"])*megatabulka["quantity"]
print(megatabulka[["quantity","cena_za_kus_bez_dph","naklad","hruby_zisk"]])
print(megatabulka["hruby_zisk"].sum())  # sum() přeskakuje NaN, takže vyřazené zboží se do součtu nepočítá
megatabulka["cena_za_kus_s_dph"]=megatabulka["cena_za_kus_bez_dph"]*1.21  # DPH 21 % = vynásobit číslem 1,21
print(megatabulka[["cena_za_kus_bez_dph","cena_za_kus_s_dph"]])

# ---------------------------------------------------------------
# 8) ÚPRAVA ORDER_ID: Z TEXTU NA ČÍSLO
# ---------------------------------------------------------------
print(megatabulka["order_id"])  # hodnoty vypadají jako "OBJ-000123" (text)
# Řetězec kroků, čte se zleva doprava:
#   megatabulka["order_id"]   vybere sloupec
#   .str                      "otevře" textové metody pro celý sloupec najednou
#   .replace("OBJ-","")       v každé hodnotě nahradí text "OBJ-" prázdným textem (= smaže předponu): "OBJ-000123" -> "000123"
#   .astype(int)              převede text na celé číslo; úvodní nuly zmizí: "000123" -> 123

megatabulka["order_id"]=megatabulka["order_id"].str.replace("OBJ-","").astype(int)
print(megatabulka.info())  # order_id má teď typ int64
#datumové operace
# ---------------------------------------------------------------
# 9) DATUMOVÉ OPERACE (.dt)
# ---------------------------------------------------------------
megatabulka["rok_objednavky"]=megatabulka["order_date"].dt.year #2025,2026
megatabulka["mesic_objednavky"]=megatabulka["order_date"].dt.month #1-12
# to_period("M") = období "2025-03" (rok+měsíc v jednom sloupci). Typ Period se dobře řadí a seskupuje;
# na rozdíl od samotného měsíce (1–12) nesmíchá leden 2025 s lednem 2026.
megatabulka["rok_mesic_objednavky"]=megatabulka["order_date"].dt.to_period("M")
# day_of_week: 0 = pondělí ... 6 = neděle. "+1" posune číslování na 1 = pondělí ... 7 = neděle
# (komentář dole "0 - pondělí 6 - neděle" tedy platí jen pro verzi BEZ +1).
megatabulka["den_v_tydnu_objednavky"]=megatabulka["order_date"].dt.day_of_week+1 #0 - pondělí 6 - neděle
# Název dne: dt.day_of_week dá číslo 0–6 a map({číslo: název}) ho přeloží podle slovníku.
megatabulka["den_v_tydnu_nazev_objednavky"]=(megatabulka["order_date"]
                                             .dt.day_of_week
                                             .map({0:"Pondeli",1:"Utery",2:"Streda", 3:"Ctvrtek", 4:"Patek", 5: "Sobota", 6:"Nedele"}))
print(megatabulka["den_v_tydnu_nazev_objednavky"])
print(megatabulka[["order_date","registrace"]])

# Odečtením dvou datumů vznikne "Timedelta" (časový úsek); .dt.days z něj vezme jen celé dny.
# U hostů je registrace NaT (prázdné datum), takže i rozdíl je NaN.
# Kontrola, kterou stojí za to přidat: nikdo nesmí nakoupit PŘED registrací (rozdíl >= 0):
#   assert (megatabulka["rozdil_v_datech"].dropna() >= 0).all()
megatabulka["rozdil_v_datech"]=(megatabulka["order_date"]-megatabulka["registrace"]).dt.days
print(megatabulka["rozdil_v_datech"])

# ---------------------------------------------------------------
# 10) DOKONČENÉ OBJEDNÁVKY A SOUČTY PO MĚSÍCÍCH
# ---------------------------------------------------------------
# tržby z dokončených objednavek
print(megatabulka.info())
# Dva kroky v jednom řádku: [megatabulka["status"]=="dokonceno"] ponechá jen řádky se stavem "dokonceno"
# (vrácené a zrušené se nepočítají), ["hruby_zisk"] vybere sloupec, .sum() ho sečte.
print(megatabulka[megatabulka["status"]=="dokonceno"]["hruby_zisk"].sum())

# Samostatná tabulka jen s dokončenými objednávkami, ať se filtr nemusí opakovat.
dokoncene_objednavky_mega=megatabulka[megatabulka["status"]=="dokonceno"]
#tržby z dokončených objednávek v jednotlivých měsících

# groupby rozdělí řádky do skupin a pro každou spočítá souhrn ("rozděl – spočítej – slep"):
#   groupby(["rok_objednavky","mesic_objednavky"])  skupina = každá KOMBINACE rok+měsíc (např. 2025/3)
#   ["hruby_zisk"]                                  který sloupec z každé skupiny sčítáme
#   .sum()                                          součet za skupinu
# Výsledek je řada s dvouúrovňovým indexem (MultiIndex): rok a měsíc. Skupiny jsou seřazené vzestupně.
# Kontrola: součet této řady musí být roven součtu z předchozího výpisu (jinak groupby zahodil nějaké řádky).
trzby_dle_mesice=dokoncene_objednavky_mega.groupby(["rok_objednavky", "mesic_objednavky"])["hruby_zisk"].sum()
print(trzby_dle_mesice)

# describe() = souhrn čísel: count (počet NEprázdných), mean (průměr), std (rozptyl hodnot kolem průměru),
# min, 25 % / 50 % / 75 % (kvartily; 50 % = medián) a max.
# count je menší než počet řádků o řádky, kde je zisk NaN (zboží bez nákladové ceny v katalogu).
print(megatabulka["hruby_zisk"].describe())

# ---------------------------------------------------------------
# 11) NP.WHERE: ROZDĚLENÍ ŘÁDKŮ NA "VELKÁ" A "BĚŽNÁ"
# ---------------------------------------------------------------
import numpy as np  # numpy = matematika nad celými sloupci; běžně se importy dávají úplně nahoru
# np.where(podmínka, hodnota_když_platí, hodnota_když_neplatí) = IF z Excelu pro celý sloupec najednou.
# Hranice 750 jsme zvolili my – není v datech, je to rozhodnutí. Dá se ověřit proti výstupu describe() výše.
megatabulka["velikost_objednavky"]=np.where(megatabulka["hruby_zisk"]>750,"velka","bezna")
print(megatabulka[["hruby_zisk","velikost_objednavky"]])

# ---------------------------------------------------------------
# 12) GROUPBY PO OBJEDNÁVKÁCH
# ---------------------------------------------------------------
# trzby dle objednavek a roku+mesic

# Skupina = KOMBINACE order_id + rok_mesic. Protože jedna objednávka patří jen do jednoho měsíce, je to ve skutečnosti
# skupina = jedna objednávka: sečtou se její řádky (1–4 položky) do jednoho čísla. Tohle je důležitý krok
# "z řádků na objednávky": tabulka měla jeden řádek na položku, výsledek má jeden řádek na objednávku.
# Díky tomu lze počítat třeba "průměrná hodnota objednávky" (mean z řádků by dal průměr POLOŽKY!).
# Kontrola: součet této řady = součet hruby_zisk dokončených objednávek, a počet řádků = počet unikátních order_id:
#   assert len(trzby_dle_mesice_obj) == dokoncene_objednavky_mega["order_id"].nunique()
trzby_dle_mesice_obj=dokoncene_objednavky_mega.groupby(["order_id", "rok_mesic_objednavky"])["hruby_zisk"].sum()
print(trzby_dle_mesice_obj)
# Series.info() funguje až od pandas 1.4; ve starší verzi by tady byla chyba AttributeError.
print(trzby_dle_mesice_obj.info())

# ---------------------------------------------------------------
# 13) ZÁKAZNÍCI PODLE SEGMENTU A GRAF
# ---------------------------------------------------------------
# Pozor, tentokrát se groupby dělá na tabulce zakaznici (1 řádek = 1 zákazník), ne na megatabulce.
# Výsledek je tedy počet ZÁKAZNÍKŮ v každém segmentu, ne počet objednávek.
#   groupby("segment")   skupina = jeden segment (např. Premium, Standard...)
#   ["registrace"]       libovolný sloupec, který každý zákazník má; jde jen o to, co se počítá
#   .count()             počet NEprázdných hodnot ve skupině (kdyby někdo neměl registraci, nezapočítal by se)
pomocna_tabulka=zakaznici.groupby("segment")["registrace"].count()
print(pomocna_tabulka)

import matplotlib.pyplot as plt  # matplotlib = kreslení grafů; "plt" je běžná zkratka
# subplots() vytvoří plátno (fig) a v něm osy grafu (ax); figsize = (šířka, výška) v palcích
fig, ax = plt.subplots(figsize=(16,9))
# ax.bar(x, výška) nakreslí sloupcový graf:
#   pomocna_tabulka.index   = popisky osy X (názvy segmentů – po groupby jsou segmenty v INDEXU)
#   pomocna_tabulka.values  = výšky sloupců (počty zákazníků)
ax.bar(pomocna_tabulka.index, pomocna_tabulka.values)
plt.show()  # zobrazí graf (program čeká, dokud okno nezavřeš)
