# =====================================================================================
# VIZUALIZACE DAT II – CO JSME STIHLI V HODINĚ 16. 11. 2026 · OKOMENTOVANÝ SKRIPT
# Skillmea · lektor: Garik · dataset: Vlnka Shop (čistá verze)
# =====================================================================================

import pandas as pd                                          # pandas = tabulky (DataFrame)
import numpy as np                                           # numpy = matematika nad celými sloupci
import matplotlib.pyplot as plt                              # matplotlib = základ všech grafů v Pythonu
import matplotlib.colors as mcolors                          # nástroje pro barvy (v hodině jsme ho zatím nepoužili; přijde u mapy krajů)
import seaborn as sns                                        # seaborn = nadstavba nad matplotlib: heatmapy, boxploty, violiny jedním řádkem
#pip install seaborn                                         # když seaborn nemáte: tenhle příkaz spustit v terminálu (ne v Pythonu)

import os                                                    # práce se složkami
from sklearn.model_selection import train_test_split         # rozdělení na train a test
from sklearn.linear_model import LogisticRegression          # model ano/ne
from sklearn.metrics import roc_auc_score                    # AUC = jak dobře model řadí zákazníky
from sklearn.metrics import roc_curve                        # NOVĚ: body ROC křivky (fpr, tpr) pro kreslení (použijeme až příště)
from sklearn.metrics import confusion_matrix                 # NOVĚ: matice záměn (kolik tipů trefilo, kolik minulo) (až příště)
os.makedirs("outputs", exist_ok=True)                        # složka pro obrázky; když už existuje, nic se nestane (exist_ok=True)

# =====================================================================================
# 0) STYL GRAFŮ
# =====================================================================================
MODRA="#2a78d6"
SVETLA_MODRA="#a9c8f0"
ORANZOVA="#eb6834"
SEDA="#52514e"
SVETLE_SEDA="#c9c8c1"

BARVY_SEGMENT={"Standard":MODRA, "Premium":ORANZOVA, "Firemní":"#2f8f6b"}
PORADI_SEGMENT=["Standard","Premium","Firemní"]              # pevné pořadí segmentů v grafech

plt.rcParams.update({"axes.spines.top":False, "axes.spines.right":False,   # rcParams = nastavení vzhledu pro VŠECHNY další grafy; bez horního a pravého rámečku
                     "axes.grid":True, "grid.color":"#e6e5e1", "grid.linewidth":0.8,   # zapnout světlou mřížku
                     "axes.axisbelow":True,                  # mřížka leží POD daty (ne přes ně)
                     "axes.edgecolor":"#c3c2b7", "axes.labelcolor":SEDA,   # barva os a popisků os
                     "xtick.color":SEDA, "ytick.color":SEDA, "font.size":10,   # barva čísel na osách a velikost písma
                     "axes.titleweight":"bold", "axes.titlesize":11})   # titulky tučně a o něco větší, aby je oko našlo hned

# =====================================================================================
# 1) NAČTENÍ A PŘÍPRAVA DAT – z minula
# =====================================================================================

objednavky=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\objednavky2.csv")
produkty=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\produkty2.csv")
zakaznici=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\zakaznici2.csv")

objednavky["order_date"]=pd.to_datetime(objednavky["order_date"], format="%Y-%m-%d %H:%M")

op=objednavky.merge(produkty, on="product_id", how="left", validate="m:1")   # ke každé položce přidá kategorii a cenu; how="left" = všechny řádky objednávek zůstanou; validate="m:1" = pojistka: produkt je v katalogu jen 1×
assert len(op)==len(objednavky)                              # KONTROLA: merge nezměnil počet řádků (44 906); kdyby ano, assert zastaví program s chybou
op["trzba_bez_dph"]=op["quantity"]*op["cena_za_kus_bez_dph"]*(1-op["discount_pct"]/100)   # tržba řádku = množství × cena × (1 − sleva v procentech/100)

agg_tabulka=(op                                               # závorky ( ) umožní psát každou metodu řetězce na nový řádek
             .groupby("order_id")                            # 1 skupina = 1 objednávka (objednávka má víc řádků-položek)
             .agg(order_date=("order_date","first"),         # datum objednávky (u všech řádků objednávky stejné, stačí první)
                  customer_id=("customer_id","first"),       # zákazník
                  status=("status","first"),                 # dokonceno / vraceno / zruseno
                  channel=("channel","first"),               # web / mobilni_app / telefon
                  dopravce=("dopravce","first"),             # NOVĚ dopravce (heatmapa doručení)
                  delivery_days=("delivery_days","first"),   # NOVĚ doba doručení ve dnech
                  trzba_bez_dph=("trzba_bez_dph","sum"))     # tržba objednávky = součet tržeb jejích položek
             .reset_index()                                  # order_id se po groupby stal indexem; reset_index ho vrátí zpět do sloupce
             .sort_values(["order_date","order_id"]))        # seřadit podle času (a při shodě podle čísla objednávky)
print("Objednávek:", len(agg_tabulka))                       # KONTROLA: 26 103
print(agg_tabulka.head())                                    # prvních 5 řádků tabulky (.head() = "hlava" tabulky)


dok=agg_tabulka[agg_tabulka["status"]=="dokonceno"].copy()   # jen dokončené objednávky (vrácené a zrušené do tržby nepatří); .copy() = samostatná kopie, můžeme do ní bezpečně přidávat sloupce
dok=dok.merge(zakaznici[["customer_id","segment","kraj","zdroj"]], on="customer_id", how="left", validate="m:1")   # přidáme 3 sloupce o zákazníkovi; v zakaznici bereme jen potřebné sloupce
assert len(dok)==23940                                       # KONTROLA: dokončených objednávek 23 940
dok["rok"]=dok["order_date"].dt.year                         # rok objednávky (2025 / 2026); .dt
dok["mesic"]=dok["order_date"].dt.month                      # číslo měsíce 1–12
dok["rok_mesic"]=dok["order_date"].dt.to_period("M")         # rok-měsíc (např. 2026-03) pro časové osy; "M" = měsíční období
dok["den_v_tydnu"]=dok["order_date"].dt.dayofweek            # 0 = pondělí ... 6 = neděle
dok["hodina"]=dok["order_date"].dt.hour                      # hodina objednávky 0–23


# ---- model z lekce 3 (beze změny; počítá se tady jen proto, abychom měli skóre zákazníků) ----
hranice=pd.Timestamp("2026-06-30 23:59")                     # "dnes" z pohledu modelu
konec_okna=hranice+pd.Timedelta(days=123)                    # konec okna cíle (123 dní po hranici = 31. 10. 2026)
obchody_historie=dok[dok["order_date"]<=hranice]             # z historie budou PŘÍZNAKY
obchody_budoucnost=dok[(dok["order_date"]>hranice)&(dok["order_date"]<=konec_okna)]

zakaznik=pd.DataFrame(index=sorted(obchody_historie["customer_id"].unique()))   #řádek = zákazník, který v historii nakoupil; unique = každé id jednou; sorted = seřadit
zakaznik.index.name="customer_id"                            # pojmenovat index
zakaznik["koupi_znovu"]=zakaznik.index.isin(obchody_budoucnost["customer_id"]).astype(int)   # cíl: je zákazník v budoucím okně? isin -> True/False; astype(int) -> 1/0
print("Zákazníků:", len(zakaznik), "| podíl koupí znovu:", round(zakaznik["koupi_znovu"].mean(),3))

nakupy=(obchody_historie.groupby("customer_id")               # RFM tabulka: skupina = zákazník (viz lekce 3)
        .agg(posledni_nakup=("order_date","max"), frekvence=("order_id","count"), monetary=("trzba_bez_dph","sum")))   # kdy naposledy, kolik objednávek, kolik celkem utratil
print(nakupy.head())
zakaznik=zakaznik.join(nakupy)                               # připojit podle indexu (customer_id)
zakaznik["recency_dny"]=(hranice-zakaznik["posledni_nakup"]).dt.days   # dny od posledního nákupu; .dt.days = z časového rozdílu jen počet dnů
zakaznik["AVG_hodnota_objednavky"]=zakaznik["monetary"]/zakaznik["frekvence"]   # průměrná objednávka
zakaznik["monetary_log"]=np.log1p(zakaznik["monetary"])      # log útraty: log1p = log(1+x); zmenší dlouhý pravý ocas

h=obchody_historie.sort_values(["customer_id","order_date"]).copy()   # seřadit podle zákazníka a času (nutné pro diff)
h["interval"]=h.groupby("customer_id")["order_date"].diff().dt.days   # dny od předchozí objednávky téhož zákazníka (první objednávka = NaN)
zakaznik["interval_prumer"]=h.groupby("customer_id")["interval"].mean()   # obvyklý odstup mezi nákupy (zákazník s 1 nákupem bude NaN -> doplníme mediánem)

posledni90=obchody_historie[obchody_historie["order_date"]>hranice-pd.Timedelta(days=90)]   # objednávky z posledních 90 dní před hranicí
zakaznik["objednavek_90d"]=posledni90.groupby("customer_id").size().reindex(zakaznik.index, fill_value=0)   # počet objednávek; reindex přidá zákazníky, co v 90 dnech nenakoupili, s hodnotou 0

vsechny=agg_tabulka[agg_tabulka["order_date"]<=hranice]      # všechny stavy (i vrácené a zrušené) do hranice
zakaznik["podil_vracenych"]=(vsechny["status"]!="dokonceno").groupby(vsechny["customer_id"]).mean().reindex(zakaznik.index)   # podíl vrácených/zrušených: True/False průměr = podíl; groupby podle zákazníka   # podíl vrácených/zrušených

segment=zakaznici.set_index("customer_id")["segment"].reindex(zakaznik.index)   # segment každého zákazníka ve stejném pořadí jako tabulka zakaznik
zakaznik["segment_premium"]=(segment=="Premium").astype(int)   # one-hot: 1 = Premium, jinak 0
zakaznik["segment_firemni"]=(segment=="Firemní").astype(int)   # one-hot: 1 = Firemní, jinak 0 (Standard = oba sloupce 0)

y=zakaznik["koupi_znovu"]                                    # cíl
X=zakaznik.drop(columns=["koupi_znovu","posledni_nakup","monetary"])   # příznaky = vše kromě cíle, data posledního nákupu (datum model nezpracuje) a surové útraty (máme log)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)   # 80 % train / 20 % test; random_state = stejné rozdělení při každém spuštění; stratify = stejný podíl cíle v obou částech
median=X_train.median()                                      # medián jen z trainu
X_train=X_train.fillna(median)                               # doplnit chybějící hodnoty v trainu
X_test=X_test.fillna(median)                                 # doplnit test čísly z TRAINU (test nesmí ovlivnit nic z učení)
prumer=X_train.mean()                                        # průměr z trainu
odchylka=X_train.std()                                       # směrodatná odchylka z trainu
X_train_s=(X_train-prumer)/odchylka                          # škálovaný train (z-skóre)
X_test_s=(X_test-prumer)/odchylka                            # škálovaný test (čísly z trainu)

model=LogisticRegression(max_iter=1000)                      # model; max_iter = kolikrát smí zkoušet vylepšit váhy
model.fit(X_train_s, y_train)                                # učí se jen na trainu
p_test=model.predict_proba(X_test_s)[:,1]                    # pravděpodobnost nákupu pro testovací zákazníky; [:,1] = sloupec "koupí" (sloupec 0 = "nekoupí")
print("AUC modelu:", round(roc_auc_score(y_test, p_test),3))

X_vse=X.fillna(median)                                       # všichni zákazníci (train i test), doplněno mediánem z trainu
zakaznik["pravdepodobnost"]=model.predict_proba((X_vse-prumer)/odchylka)[:,1]   # skóre pro všechny zákazníky
zakaznik["riziko"]=pd.cut(zakaznik["pravdepodobnost"], bins=[0,0.4,0.7,1], labels=["vysoké riziko","střední","věrný"])   # pd.cut = rozřazení čísel do přihrádek podle hranic 0–0,4–0,7–1
zakaznik["segment"]=segment                                  # segment jako text
zakaznik["kraj"]=zakaznici.set_index("customer_id")["kraj"].reindex(zakaznik.index)
zakaznik["zdroj"]=zakaznici.set_index("customer_id")["zdroj"].reindex(zakaznik.index)
# =====================================================================================
# 2) Mřížka malých grafů se STEJNOU osou x
# =====================================================================================
radky=op[op["status"]=="dokonceno"].copy()                   # dokončené POLOŽKY (kategorie je na úrovni položky, ne objednávky, proto bereme op, ne dok)
radky["rok"]=radky["order_date"].dt.year                     # rok
radky["měsíc"]=radky["order_date"].dt.month                  # měsíc 1–12 (sloupec s diakritikou funguje, jen pozor: v tabulce dok se stejná věc jmenuje "mesic" bez háčku)
print(radky.info())                                          # přehled sloupců a typů; POZOR: .info() se vypíše sama a vrací None, proto za výpisem uvidíš ještě řádek "None". Stačí psát jen radky.info()
kat_mesic=radky.pivot_table(index=["kategorie","měsíc"],     # řádky: kategorie a v ní měsíc 1–12 (dvouúrovňový index)
                            columns="rok",                   # sloupce: 2025 a 2026 (jedna čára na rok)
                            values="trzba_bez_dph",          # co sčítáme
                            aggfunc="sum")/1e6               # součet tržby; /1e6 = v milionech Kč
print(kat_mesic)
kat_poradi=(radky[radky["kategorie"]!="Vyřazené zboží"].groupby("kategorie")["trzba_bez_dph"].sum()   # "Vyřazené zboží" je mimo katalog, do grafů ho nedáváme; tržba každé kategorie
            .sort_values(ascending=False).index.to_list())   # seřadit od největší; .index = názvy kategorií; .to_list() = obyčejný seznam
print(kat_poradi)


# ---- 2.1 Small multiples: čárový graf tržeb pro každou kategorii zvlášť ----
fig, osy=plt.subplots(2,3, figsize=(12,6))      # fig = celé plátno; osy = pole 2×3 os (jedna osa = jeden malý graf); figsize = šířka × výška v palcích
osy=osy.flatten() #na projití jedním cyklem                # pole 2×3 -> seznam šesti os, aby šly projít jedním cyklem

for i, kat in enumerate(kat_poradi):                         # i = 0..5 (pořadí panelu), kat = název kategorie; enumerate dá obojí zároveň
    ax=osy[i]                                                # panel pro tuto kategorii
    cast=kat_mesic.loc[kat] #tvorba indexů -> měsíc 1-12, sloupce 2025/2026   # řádky jen této kategorie: index = měsíc 1–12, sloupce = 2025 a 2026
    ax.plot(cast.index, cast[2025], color=SVETLE_SEDA, label="2025")   # loni = šedá (pozadí, kontext); x = měsíce, y = tržba
    ax.plot(cast.index, cast[2026], color=MODRA, label="2026")         # letos = barva (hlavní řada); chybějící listopad a prosinec se prostě nenakreslí
    ax.set_title(kat, loc="left")                            # titulek panelu = název kategorie; loc="left" = zarovnaný doleva
    ax.set_xticks(range(1,13))                               # popisky osy x: 1..12 (range(1,13) = čísla 1 až 12, 13 už ne)
osy[0].legend(frameon=False)                                 # legenda stačí v prvním panelu (platí pro všechny); frameon=False = bez rámečku
fig.supxlabel("Měsíc v roce")                                # společný popisek osy x pro celé plátno
fig.supylabel("Tržby v milionech Kč")                        # společný popisek osy y pro celé plátno
fig.tight_layout()                                           # srovná mezery mezi panely, aby se nic nepřekrývalo
fig.savefig("outputs/2026-16-11 male liniove grafy.png")     # uložit obrázek do složky outputs; vždy PŘED plt.show()
plt.show()                                                   # zobrazit graf (po zavření okna program pokračuje)

# ---- 2.2 Sezónní index (graf 2): všechny kategorie se srovnají na jednu stupnici ----
fig, osy=plt.subplots(2,3, figsize=(12,6))
osy=osy.flatten()                                            # pole os -> seznam šesti os
for i, kat in enumerate(kat_poradi):                         # cyklus přes 6 kategorií
    ax = osy[i]                                              # panel pro tuto kategorii
    cast = kat_mesic.loc[kat][2025]  # tvorba indexů -> měsíc 1-12, sloupce 2025/2026   # tržba kategorie po měsících, jen rok 2025 (jediný rok s celými 12 měsíci); výsledek = řada s 12 hodnotami
    index=cast/cast.mean()                                   # sezónní index = měsíc / roční průměr kategorie
    ax.axhline(1,color=SVETLE_SEDA, linestyle="--")          # vodorovná čárkovaná čára na hodnotě 1 = "normální měsíc"; axhline = horizontální čára přes celý graf
    ax.plot(index.index, index.values, color=MODRA, marker="o")   # tvar sezónnosti; marker="o" = kolečko v každém měsíci; .index = měsíce (osa x), .values = indexy (osa y)
    ax.set_title(kat, loc="left")                            # titulek panelu = název kategorie
    ax.set_xticks(range(1, 13))                              # popisky osy x: 1..12
fig.supxlabel("Měsíc v roce")                                # společný popisek osy x
fig.tight_layout()                                           # srovná mezery
plt.show()                                                   # zobrazit graf

# ---- 2.3 Zvýraznění jedné kategorie (graf 3): šedý kontext + jedna barevná čára ----
fig, ax=plt.subplots(figsize=(12,6))
for kat in kat_poradi:                                       # nejdřív všechny kategorie šedě (kontext)
    rada=kat_mesic.loc[kat][2025]                            # tržba kategorie po měsících v roce 2025
    ax.plot(rada.index, rada/rada.mean(), color=SVETLE_SEDA) # sezónní index kategorie, světle šedě
rada=kat_mesic.loc["Hračky"][2025]                           # jedna kategorie, o které vyprávíme
ax.plot(rada.index, rada/rada.mean(), color=ORANZOVA, label="Hračky")   # barva jen pro ni; nakreslená POSLEDNÍ, takže leží nad šedými čarami

ax.set_xlabel("Měsíc v roce")                                # popisek osy x
ax.set_ylabel("Sezonní index -> 1 průměrný měsíc")           # popisek osy y: co číslo znamená (POZOR: překlep "Sezonní" -> správně "Sezónní")
ax.legend(frameon=False)                                     # legenda jen s položkou "Hračky" (šedé čáry nemají label)
fig.tight_layout()                                           # srovná mezery
plt.show()                                                   # zobrazit graf

# =====================================================================================
# 3) HEATMAPY – tabulka převedená na barvy
# =====================================================================================
# ---- 3.1 Kdy lidé objednávají? Den v týdnu × hodina – nejdřív "ručně" přes matplotlib ----
# 1. matplotlib
tab_dh=pd.crosstab(dok["den_v_tydnu"], dok["hodina"])        # crosstab = počet objednávek pro každou kombinaci (den, hodina); matice 7 řádků × 24 sloupců
tab_dh.index=["Po","Út", "St", "Čt", "Pá", "So", "Ne"]       # čísla dnů 0–6 přejmenujeme na zkratky (pořadí = 0 je pondělí)

fig, ax=plt.subplots(figsize=(12,6))
obrazek=ax.imshow(tab_dh.values,                             # imshow = "zobraz matici jako obrázek"; .values = čistá čísla bez popisků řádků a sloupců
                  cmap="Blues")                              # cmap = paleta; "Blues" = JEDNA barva od světlé po tmavou = množství (sequential)
ax.set_xticks(range(24))                                     # 24 značek na ose x (jedna na hodinu)
ax.set_xticklabels(tab_dh.columns)                           # popisky značek = hodiny 0–23 (názvy sloupců crosstabu)
ax.set_yticks(range(7))                                      # 7 značek na ose y (jedna na den)
ax.set_yticklabels(tab_dh.index)                             # popisky značek = zkratky dnů
fig.colorbar(obrazek, ax=ax)                                 # barevná stupnice = "legenda" heatmapy: BEZ NÍ barvy nic neznamenají
plt.show()                                                   # zobrazit graf

# ---- 3.2 Totéž jedním příkazem v seaborn ----
fig, ax=plt.subplots(figsize=(12,6))
sns.heatmap(tab_dh,                                          # matice (seaborn si sám vezme popisky z indexu a sloupců)
            cmap="Blues",                                    # stejná paleta jako výše
            cbar_kws={"label": "Počet objednávek"},          # popisek barevné stupnice (cbar = colorbar; kws = slovník nastavení)
            ax=ax                                            # kam kreslit
)
plt.show()                                                   # zobrazit graf

# ---- 3.3 Web × mobilní aplikace ----
fig, osy=plt.subplots(1,2, figsize=(12,6))
for ax, kanal in zip(osy,["web", "mobilni_app"]):            # zip spáruje osu a název kanálu: (osy[0],"web"), (osy[1],"mobilni_app")
    cast=dok[dok["channel"]==kanal]                          # objednávky jednoho kanálu
    tab=pd.crosstab(cast["den_v_tydnu"], cast["hodina"],normalize="all")*100   # normalize="all" = každé políčko jako podíl na CELÉM kanálu (součet = 1); *100 = procenta
    tab.index = ["Po", "Út", "St", "Čt", "Pá", "So", "Ne"]   # čísla dnů 0–6 -> zkratky
    sns.heatmap(tab,cmap="Blues",cbar_kws={"label": kanal}, ax=ax)   # heatmapa kanálu; popisek stupnice je zatím název kanálu
plt.show()                                                   # zobrazit oba grafy

# ---- 3.4 Příběh dopravců: průměrná doba doručení, dopravce × měsíc ----
print(dok.info())
dop_mesic=dok.pivot_table(index="dopravce", columns="rok_mesic",values="delivery_days", aggfunc="mean")   # matice dopravce × měsíc; v buňce průměr dní doručení
dop_mesic.columns=dop_mesic.columns.astype(str)              # období (např. 2025-12) -> text, ať seaborn popisky sloupců umí zobrazit

fig, ax=plt.subplots(figsize=(12,6))
sns.heatmap(dop_mesic,cmap="YlOrRd",                         # paleta žlutá -> oranžová -> červená = "čím víc, tím horší" (u doby doručení je víc = špatně)
            annot=True,                                      # vypsat číslo do každého políčka
            fmt=".1f",                                       # formát čísla: 1 desetinné místo
            cbar_kws={"label": "Průměr dní doručení"}, ax=ax)   # popisek stupnice; kam kreslit
ax.grid(False)                                               # vypnout mřížku
plt.show()                                                   # zobrazit graf

# =====================================================================================
# SHRNUTÍ: CO SI ZE DNEŠKA ODNÉST (3 věty pro studenty)
# =====================================================================================
# 1) Mnoho skupin v čase -> small multiples (mřížka malých grafů). Chceš srovnávat velikosti? Společná osa. Chceš srovnávat tvary? Vydělit vlastním průměrem (index).
# 2) Dvě kategorie na osách + číslo v buňce -> heatmapa. Postup: crosstab / pivot_table -> paleta -> sns.heatmap. Vždy popisek barevné stupnice a ax.grid(False).
# 3) Jedna barva = množství (Blues, YlOrRd). Při porovnání dvou heatmap vedle sebe MUSÍ mít stejné vmin a vmax.
