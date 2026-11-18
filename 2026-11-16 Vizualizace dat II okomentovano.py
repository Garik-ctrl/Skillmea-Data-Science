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
# SHRNUTÍ: CO SI ZE DNEŠKA ODNÉST
# =====================================================================================
# 1) Mnoho skupin v čase -> small multiples (mřížka malých grafů). Chceš srovnávat velikosti? Společná osa. Chceš srovnávat tvary? Vydělit vlastním průměrem (index).
# 2) Dvě kategorie na osách + číslo v buňce -> heatmapa. Postup: crosstab / pivot_table -> paleta -> sns.heatmap. Vždy popisek barevné stupnice a ax.grid(False).
# 3) Jedna barva = množství (Blues, YlOrRd). Při porovnání dvou heatmap vedle sebe MUSÍ mít stejné vmin a vmax.


# #####################################################################################
# ###################   TADY JSME V HODINĚ 16. 11. SKONČILI   #########################
# #####################################################################################


fig, osy=plt.subplots(2,3, figsize=(12,6), sharex=True, sharey=True)
osy=osy.flatten()
for i, kat in enumerate(kat_poradi):
    ax=osy[i]
    cast=kat_mesic.loc[kat][2025]
    index=cast/cast.mean()
    ax.axhline(1, color=SEDA, linewidth=1, linestyle="--")
    ax.plot(index.index, index.values, color=MODRA, linewidth=2, marker="o", markersize=3)
    ax.set_title(kat, loc="left")
    ax.set_xticks(range(1,13))
fig.supxlabel("Měsíc v roce 2025")
fig.supylabel("Sezónní index (1 = průměrný měsíc)")
fig.suptitle("Po vydělení průměrem jsou kategorie srovnatelné: nejvýraznější sezónu mají Hračky (prosinec)", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/v2_02_small_multiples_index.png", dpi=120)
plt.show()

fig, ax=plt.subplots(figsize=(9,4.5))
for kat in kat_poradi:
    rada=kat_mesic.loc[kat][2025]
    ax.plot(rada.index, rada/rada.mean(), color=SVETLE_SEDA, linewidth=1.5)
rada=kat_mesic.loc["Hračky"][2025]
ax.plot(rada.index, rada/rada.mean(), color=ORANZOVA, linewidth=3)
ax.text(12.1, (rada/rada.mean()).iloc[-1], "Hračky", color=ORANZOVA, va="center", fontweight="bold")
ax.set_xticks(range(1,13))
ax.set_xlabel("Měsíc v roce 2025"); ax.set_ylabel("Sezónní index (1 = průměrný měsíc)")
ax.set_title("Hračky mají v prosinci 3× průměrnou měsíční tržbu, ostatní kategorie jsou klidnější", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_03_zvyrazneni_jedne.png", dpi=120)
plt.show()

fig, osy=plt.subplots(1, 2, figsize=(14,3.6), sharey=True)
for ax, kanal in zip(osy, ["web","mobilni_app"]):
    cast=dok[dok["channel"]==kanal]
    tab=pd.crosstab(cast["den_v_tydnu"], cast["hodina"], normalize="all")*100
    tab.index=["Po","Út","St","Čt","Pá","So","Ne"]
    sns.heatmap(tab, cmap="Blues", vmin=0, vmax=1.5,
                linewidths=0.3, cbar=(kanal=="mobilni_app"),
                cbar_kws={"label":"% objednávek kanálu"}, ax=ax)
    ax.grid(False)
    ax.set_title(kanal, loc="left"); ax.set_xlabel("Hodina"); ax.set_ylabel("")
fig.suptitle("Web i aplikace mají stejný rytmus: špička 19–21 h, ve dne rovnoměrně", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/v2_06_heatmapa_kanaly.png", dpi=120)
plt.show()

dok["mesic_cislo"]=dok["order_date"].dt.year*12+dok["order_date"].dt.month
prvni_cislo=dok.groupby("customer_id")["mesic_cislo"].min()
prvni_text=dok.groupby("customer_id")["order_date"].min().dt.strftime("%Y-%m")
dok["kohorta"]=dok["customer_id"].map(prvni_text)
dok["mesic_od_prvniho"]=dok["mesic_cislo"]-dok["customer_id"].map(prvni_cislo)
kohorty=dok.groupby(["kohorta","mesic_od_prvniho"])["customer_id"].nunique().unstack()
retence=kohorty.div(kohorty[0], axis=0)
retence_graf=retence.drop(index="2025-01")
retence_graf=retence_graf.iloc[:, 1:13].dropna(how="all")
retence_graf.index=retence_graf.index+" (n="+kohorty.loc[retence_graf.index,0].astype(int).astype(str)+")"

fig, ax=plt.subplots(figsize=(12,7.5))
sns.heatmap(retence_graf, cmap="Blues", vmin=0, vmax=0.6, annot=True, fmt=".0%", annot_kws={"size":7},
            linewidths=0.5, cbar_kws={"label":"Podíl kohorty, který v daném měsíci nakoupil"}, ax=ax)
ax.grid(False)
ax.set_xlabel("Měsíců od prvního nákupu"); ax.set_ylabel("Kohorta (měsíc prvního nákupu)")
ax.set_title("V každém dalším měsíci nakoupí zhruba třetina nových zákazníků; kohorty se příliš neliší", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_08_kohortni_heatmapa.png", dpi=120)
plt.show()
print(retence_graf.mean().round(2))

fig, osy=plt.subplots(1, 2, figsize=(12,4))
osy[0].hist(dok["trzba_bez_dph"], bins=50, color=MODRA, edgecolor="white")
osy[0].set_title("Lineární osa: vidíme jen nejmenší objednávky", loc="left")
hranice_binu=np.logspace(np.log10(dok["trzba_bez_dph"].min()), np.log10(dok["trzba_bez_dph"].max()), 50)
osy[1].hist(dok["trzba_bez_dph"], bins=hranice_binu, color=MODRA, edgecolor="white")
osy[1].set_xscale("log")
osy[1].set_title("Logaritmická osa: vidíme celé rozdělení", loc="left")
for ax in osy:
    ax.set_xlabel("Hodnota objednávky (Kč bez DPH)"); ax.set_ylabel("Počet objednávek")
fig.tight_layout()
fig.savefig("outputs/v2_09_histogram_log.png", dpi=120)
plt.show()

fig, ax=plt.subplots(figsize=(8,5))
sns.boxplot(data=dok, x="segment", y="trzba_bez_dph", order=PORADI_SEGMENT,
            color=SVETLA_MODRA, fliersize=2, ax=ax)
ax.set_yscale("log")
prumery=dok.groupby("segment")["trzba_bez_dph"].mean().reindex(PORADI_SEGMENT)
ax.scatter(range(3), prumery.values, marker="D", color=ORANZOVA, s=60, zorder=3, label="průměr")
ax.legend(frameon=False, loc="upper left")
ax.set_xlabel(""); ax.set_ylabel("Hodnota objednávky (Kč bez DPH, log osa)")
ax.set_title("Firemní objednávky jsou typicky 4× větší; průměr (◆) tahá nahoru dlouhý ocas", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_10_boxplot_segment.png", dpi=120)
plt.show()
print(dok.groupby("segment")["trzba_bez_dph"].agg(["median","mean"]).round(0))

zakaznik["koupil_text"]=np.where(zakaznik["koupi_znovu"]==1, "koupil znovu", "nekoupil")
fig, ax=plt.subplots(figsize=(7,5))
sns.violinplot(data=zakaznik, x="koupil_text", y="recency_dny",
               order=["koupil znovu","nekoupil"],
               inner="quartile",
               cut=0,
               palette={"koupil znovu":MODRA, "nekoupil":ORANZOVA},
               hue="koupil_text", legend=False,
               ax=ax)
ax.set_xlabel(""); ax.set_ylabel("Dny od posledního nákupu (recency)")
ax.set_title("Kdo koupil znovu, nakoupil nedávno", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_11_violin_recency.png", dpi=120)
plt.show()

fig, ax=plt.subplots(figsize=(8,4.5))
sns.ecdfplot(data=dok, x="trzba_bez_dph", hue="segment",
             hue_order=PORADI_SEGMENT, palette=BARVY_SEGMENT,
             log_scale=True, linewidth=2.5, ax=ax)
ax.axhline(0.5, color=SEDA, linewidth=1, linestyle="--")
ax.set_xlabel("Hodnota objednávky (Kč bez DPH, log osa)"); ax.set_ylabel("Podíl objednávek menších než hodnota")
ax.set_title("Čára Firemních je posunutá doprava: polovina jejich objednávek přesáhne 8 600 Kč", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_12_ecdf_segment.png", dpi=120)
plt.show()

sloupce=["koupi_znovu","recency_dny","frekvence","monetary_log","AVG_hodnota_objednavky","interval_prumer","objednavek_90d","podil_vracenych"]
kor=zakaznik[sloupce].corr()
maska=np.triu(np.ones_like(kor, dtype=bool), k=1)
fig, ax=plt.subplots(figsize=(9,6.5))
sns.heatmap(kor, mask=maska,
            cmap="RdBu_r",
            vmin=-1, vmax=1, center=0,
            annot=True, fmt=".2f", annot_kws={"size":8},
            square=True,
            linewidths=0.5, cbar_kws={"shrink":0.8, "label":"korelace"}, ax=ax)
ax.grid(False)
ax.set_title("Kdo dlouho nenakoupil, ten se nevrací: koupi_znovu × recency = −0,44", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_13_korelace.png", dpi=120)
plt.show()

fig, ax=plt.subplots(figsize=(9,5.5))
bubliny=ax.scatter(zakaznik["recency_dny"], zakaznik["frekvence"],
                   s=np.sqrt(zakaznik["monetary"])/6,
                   c=zakaznik["pravdepodobnost"],
                   cmap="RdYlBu", vmin=0, vmax=1,
                   alpha=0.55, edgecolor="none")
fig.colorbar(bubliny, ax=ax, label="Pravděpodobnost nákupu (model)")
ax.set_xlabel("Dny od posledního nákupu (recency)"); ax.set_ylabel("Počet objednávek (frekvence)")
ax.set_title("Nedávno a často = šance blízko 100 %; dlouho a málo = téměř ztracený zákazník", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_14_bubliny.png", dpi=120)
plt.show()

fig, ax=plt.subplots(figsize=(9,5.5))
hexy=ax.hexbin(zakaznik["recency_dny"], zakaznik["frekvence"],
               C=zakaznik["koupi_znovu"],
               reduce_C_function=np.mean,
               gridsize=18,
               mincnt=8,
               cmap="Blues", vmin=0, vmax=1)
fig.colorbar(hexy, ax=ax, label="Podíl, kdo koupil znovu")
ax.set_xlabel("Dny od posledního nákupu (recency)"); ax.set_ylabel("Počet objednávek (frekvence)")
ax.grid(False)
ax.set_title("Mapa šancí: vlevo nahoře je koupě skoro jistá, vpravo dole skoro nikdo nekoupí", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_15_hexbin.png", dpi=120)
plt.show()

sady={"A: jen recency":["recency_dny"],
      "B: recency + frekvence + útrata":["recency_dny","frekvence","monetary_log"],
      "C: všechny příznaky":list(X_train_s.columns)}
barvy_sad=["#9a9892", "#7fb0ec", MODRA]

fig, ax=plt.subplots(figsize=(6.5,6.5))
ax.plot([0,1], [0,1], color=SEDA, linestyle="--", linewidth=1, label="náhoda (AUC 0,5)")
for (nazev, sloupce_sady), barva in zip(sady.items(), barvy_sad):
    m=LogisticRegression(max_iter=1000)
    m.fit(X_train_s[sloupce_sady], y_train)
    p=m.predict_proba(X_test_s[sloupce_sady])[:,1]
    fpr, tpr, prahy=roc_curve(y_test, p)
    auc_sady=roc_auc_score(y_test, p)
    ax.plot(fpr, tpr, color=barva, linewidth=2.5, label=f"{nazev} (AUC {auc_sady:.3f})")
    print(nazev, "AUC =", round(auc_sady,3))
tip=(p_test>=0.5)
tpr_05=(tip & (y_test==1)).sum()/(y_test==1).sum()
fpr_05=(tip & (y_test==0)).sum()/(y_test==0).sum()
ax.scatter([fpr_05], [tpr_05], color=ORANZOVA, s=70, zorder=3)
ax.annotate("práh 0,5", (fpr_05, tpr_05), xytext=(fpr_05+0.08, tpr_05-0.10), color=ORANZOVA, arrowprops={"arrowstyle":"->","color":ORANZOVA})
ax.set_xlabel("FPR: podíl nekupců, které model omylem označil za kupce"); ax.set_ylabel("TPR: podíl kupců, které model správně našel")
ax.set_aspect("equal")
ax.legend(frameon=False, loc="lower right")
ax.set_title("Samotná recency dá AUC 0,77; další příznaky ho zvednou jen o 0,02", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_16_roc.png", dpi=120)
plt.show()

koef=pd.Series(model.coef_[0], index=X_train_s.columns).sort_values()
barvy=np.where(koef>0, MODRA, ORANZOVA)
fig, ax=plt.subplots(figsize=(8,5))
ax.barh(koef.index, koef.values, color=barvy, height=0.65)
ax.axvline(0, color=SEDA, linewidth=1)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Váha příznaku v modelu (příznaky jsou standardizované, váhy jsou srovnatelné)")
nejvetsi=koef.abs().sort_values(ascending=False).index[0]
ax.set_title(f"Nejsilnější příznak: {nejvetsi}\nmodrá = zvyšuje šanci na nákup, oranžová = snižuje", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_17_koeficienty.png", dpi=120)
plt.show()
print(koef.round(2))

test=pd.DataFrame({"skutecnost":y_test.values, "p":p_test})
test["decil"]=pd.qcut(test["p"].rank(method="first"), 10, labels=False)
decily=test.groupby("decil").agg(podil_kupcu=("skutecnost","mean"),
                                 p_prumer=("p","mean"),
                                 pocet=("skutecnost","size"))
print(decily.round(2))

fig, osy=plt.subplots(1, 2, figsize=(13,4.8))
osy[0].bar(decily.index+1, decily["podil_kupcu"], color=MODRA, width=0.7, label="skutečně koupili")
osy[0].plot(decily.index+1, decily["p_prumer"], color=ORANZOVA, marker="o", linewidth=2, label="co tipoval model")
osy[0].axhline(y_test.mean(), color=SEDA, linestyle="--", linewidth=1)
osy[0].text(0.55, y_test.mean()+0.02, "celkový průměr", color=SEDA, ha="left", va="bottom", fontsize=9)
osy[0].set_xticks(range(1,11))
osy[0].set_xlabel("Decil zákazníků podle pravděpodobnosti (1 = nejnižší, 10 = nejvyšší)"); osy[0].set_ylabel("Podíl, kdo koupil znovu")
osy[0].set_ylim(0,1.05)
osy[0].legend(frameon=False, loc="upper left")
osy[0].set_title(f"V nejlepším decilu koupí znovu {decily['podil_kupcu'].iloc[-1]:.0%}, v nejhorším jen {decily['podil_kupcu'].iloc[0]:.0%}", loc="left")
poradi_zakazniku=test.sort_values("p")
odchod=(poradi_zakazniku["skutecnost"]==0).astype(int)
zisk=odchod.cumsum().values/odchod.sum()
osloveno=np.arange(1, len(poradi_zakazniku)+1)/len(poradi_zakazniku)
osy[1].plot(osloveno, zisk, color=MODRA, linewidth=2.5, label="podle modelu")
osy[1].plot([0,1], [0,1], color=SEDA, linestyle="--", linewidth=1, label="náhodný výběr")
bod=int(0.2*len(poradi_zakazniku))
osy[1].scatter([osloveno[bod-1]], [zisk[bod-1]], color=ORANZOVA, s=70, zorder=3)
osy[1].annotate(f"20 % oslovených = {zisk[bod-1]:.0%} odcházejících", (osloveno[bod-1], zisk[bod-1]),
                xytext=(0.28, 0.12), color=ORANZOVA, arrowprops={"arrowstyle":"->","color":ORANZOVA})
osy[1].set_xlabel("Podíl nejrizikovějších zákazníků, které osloví obchodník"); osy[1].set_ylabel("Podíl odcházejících, které zachytí")
osy[1].legend(frameon=False, loc="lower right")
osy[1].set_title(f"Oslovení nejrizikovější pětiny zachytí {zisk[bod-1]:.0%} odchodů (náhodně 20 %)", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_18_decily_gain.png", dpi=120)
plt.show()

cm=confusion_matrix(y_test, (p_test>=0.5).astype(int))
fig, ax=plt.subplots(figsize=(5.5,4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            annot_kws={"size":14}, linewidths=1,
            xticklabels=["tip: nekoupí","tip: koupí"],
            yticklabels=["skutečně nekoupil","skutečně koupil"],
            ax=ax)
ax.grid(False)
ax.set_title("Matice záměn při prahu 0,5", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_19_matice_zamen.png", dpi=120)
plt.show()
tn, fp, fn, tp=cm.ravel()
print("Správně nekoupí:", tn, "| omylem koupí:", fp, "| omylem nekoupí:", fn, "| správně koupí:", tp)

print("práh | označeno jako rizikoví | z nich skutečně odešlo (přesnost) | z odcházejících zachyceno")
for prah in [0.3, 0.4, 0.5, 0.6, 0.7]:
    riziko_tip=(p_test<prah)
    zachyceno=(riziko_tip & (y_test.values==0)).sum()
    print(f"{prah:.1f}  | {riziko_tip.sum():>5}                  | {zachyceno/max(riziko_tip.sum(),1):>6.0%}                           | {zachyceno/(y_test.values==0).sum():>6.0%}")

poloha_kraju={"Karlovarský":(0,0), "Ústecký":(1,0), "Liberecký":(2,0), "Královéhradecký":(3,0),
              "Plzeňský":(0,1), "Středočeský":(1,1), "Praha":(2,1), "Pardubický":(3,1), "Olomoucký":(4,1), "Moravskoslezský":(5,1),
              "Jihočeský":(1,2), "Vysočina":(3,2), "Jihomoravský":(4,2), "Zlínský":(5,2)}

kraje=dok.groupby("kraj").agg(trzba=("trzba_bez_dph","sum"),
                              zakazniku=("customer_id","nunique"))
kraje["trzba_mil"]=kraje["trzba"]/1e6
kraje["trzba_na_zak"]=kraje["trzba"]/kraje["zakazniku"]/1e3
print(kraje.sort_values("trzba_mil", ascending=False).round(1))

fig, osy=plt.subplots(1, 2, figsize=(13,4.2))
metriky_kraju=[("trzba_mil","Tržba kraje (mil. Kč)",plt.cm.Blues), ("trzba_na_zak","Tržba na zákazníka (tis. Kč)",plt.cm.Greens)]
for ax, (sloupec, titulek, paleta) in zip(osy, metriky_kraju):
    hodnoty=kraje[sloupec]
    norm=mcolors.Normalize(vmin=hodnoty.min(), vmax=hodnoty.max())
    for kraj, (sl, rad) in poloha_kraju.items():
        hodnota=hodnoty[kraj]
        barva_pole=paleta(0.15+0.85*norm(hodnota))
        ax.add_patch(plt.Rectangle((sl, -rad), 0.94, 0.94, facecolor=barva_pole, edgecolor="white"))
        barva_textu="white" if norm(hodnota)>0.55 else SEDA
        ax.text(sl+0.47, -rad+0.55, kraj, ha="center", va="center", fontsize=7.5, color=barva_textu)
        ax.text(sl+0.47, -rad+0.25, f"{hodnota:.1f}", ha="center", va="center", fontsize=10, fontweight="bold", color=barva_textu)
    ax.set_xlim(-0.1, 6.1); ax.set_ylim(-2.1, 1.05)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(titulek, loc="left")
fig.suptitle("Absolutně vede Praha, ale na zákazníka dobře vychází i Zlínský kraj: velikost kraje a hodnota zákazníka nejsou totéž", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/v2_20_mapa_kraju.png", dpi=120)
plt.show()

kat=radky.groupby("kategorie")["trzba_bez_dph"].sum().sort_values(ascending=False)/1e6
podil_app=(dok["channel"]=="mobilni_app").groupby(dok["rok"]).mean()

fig, osy=plt.subplots(2, 3, figsize=(14,8))
osy[0,0].pie(kat.values, labels=kat.index, textprops={"fontsize":8})
osy[0,0].set_title("ŠPATNĚ: koláč se 7 výsečemi", loc="left", color=ORANZOVA)
osy[1,0].barh(kat.index[::-1], kat.values[::-1], color=MODRA, height=0.6)
for i, v in enumerate(kat.values[::-1]):
    osy[1,0].text(v, i, f" {v:.1f}", va="center", color=SEDA, fontsize=9)
osy[1,0].grid(axis="y", visible=False)
osy[1,0].set_title("LÉPE: seřazené sloupce s čísly", loc="left", color=MODRA)
osy[0,1].bar(["2025","2026"], podil_app.values, color=MODRA, width=0.5)
osy[0,1].set_ylim(0.20, 0.32)
osy[0,1].set_title("ŠPATNĚ: osa y od 0,20 – rozdíl vypadá 4×", loc="left", color=ORANZOVA)
osy[1,1].bar(["2025","2026"], podil_app.values, color=MODRA, width=0.5)
osy[1,1].set_ylim(0, 0.4)
for i, v in enumerate(podil_app.values):
    osy[1,1].text(i, v+0.01, f"{v:.0%}", ha="center", color=SEDA, fontweight="bold")
osy[1,1].set_title("LÉPE: osa od nuly – růst 1,4× je vidět, ale nepřehnaný", loc="left", color=MODRA)
osy[1,1].set_ylabel("Podíl objednávek z mobilní aplikace")
osy[0,2].imshow(tab_dh.values, cmap="jet", aspect="auto")
osy[0,2].set_title("ŠPATNĚ: duha – je žlutá 'víc' než zelená?", loc="left", color=ORANZOVA)
osy[1,2].imshow(tab_dh.values, cmap="Blues", aspect="auto")
osy[1,2].set_title("LÉPE: jedna barva, tmavší = víc", loc="left", color=MODRA)
for ax in [osy[0,2], osy[1,2], osy[0,0]]:
    ax.grid(False)
fig.suptitle("Stejná data, jiný dojem: tři nejčastější chyby", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/v2_21_anti_vzory.png", dpi=120)
plt.show()
print(podil_app.round(3))

tab_km=(radky[(radky["rok"]==2025)&(radky["kategorie"]!="Vyřazené zboží")]
        .pivot_table(index="kategorie", columns="mesic", values="trzba_bez_dph", aggfunc="sum"))
index_km=tab_km.div(tab_km.mean(axis=1), axis=0)
fig, ax=plt.subplots(figsize=(11,3.8))
sns.heatmap(index_km, cmap="RdBu_r", center=1, vmin=0, vmax=2,
            annot=True, fmt=".1f", annot_kws={"size":8}, linewidths=0.5,
            cbar_kws={"label":"Sezónní index (1 = průměrný měsíc)"}, ax=ax)
ax.grid(False)
ax.set_xlabel("Měsíc v roce 2025"); ax.set_ylabel("")
ax.set_title("Únor je nejslabší měsíc téměř všude; Hračky mají v prosinci 3× průměr", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_22_cviceni1_sezonni_heatmapa.png", dpi=120)
plt.show()

fig, ax=plt.subplots(figsize=(8,4.5))
sns.boxplot(data=dok, x="dopravce", y="delivery_days", order=["DPD","PPL","Zasilkovna","Ceska posta"],
            color=SVETLA_MODRA, fliersize=2, ax=ax)
ax.set_xlabel(""); ax.set_ylabel("Doba doručení (dny)")
ax.set_title("Česká pošta je nejpomalejší; DPD a PPL doručí obvykle za 1–3 dny", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_23_cviceni2a_doruceni_dopravce.png", dpi=120)
plt.show()

dpd=dok[dok["dopravce"]=="DPD"].copy()
dpd["rok_mesic_text"]=dpd["rok_mesic"].astype(str)
fig, ax=plt.subplots(figsize=(13,4.5))
sns.boxplot(data=dpd, x="rok_mesic_text", y="delivery_days", color=SVETLA_MODRA, fliersize=2, ax=ax)
ax.tick_params(axis="x", rotation=60)
ax.set_xlabel(""); ax.set_ylabel("Doba doručení (dny) – DPD")
ax.set_title("DPD v prosinci 2025: celý box je posunutý nahoru, nejde o pár zpožděných zásilek, ale o plošný problém", loc="left")
fig.tight_layout()
fig.savefig("outputs/v2_24_cviceni2b_dpd_mesice.png", dpi=120)
plt.show()

fig, osy=plt.subplots(1, 2, figsize=(12,4.5), sharey=True)
for ax, (nazev, sloupce_sady) in zip(osy, [("A: jen recency", ["recency_dny"]), ("C: všechny příznaky", list(X_train_s.columns))]):
    m=LogisticRegression(max_iter=1000)
    m.fit(X_train_s[sloupce_sady], y_train)
    p=m.predict_proba(X_test_s[sloupce_sady])[:,1]
    t=pd.DataFrame({"skutecnost":y_test.values, "p":p})
    t["decil"]=pd.qcut(t["p"].rank(method="first"), 10, labels=False)
    d=t.groupby("decil")["skutecnost"].mean()
    ax.bar(d.index+1, d.values, color=MODRA, width=0.7)
    ax.axhline(y_test.mean(), color=SEDA, linestyle="--", linewidth=1)
    ax.set_xticks(range(1,11)); ax.set_xlabel("Decil (1 = nejnižší pravděpodobnost)")
    ax.set_title(f"Sada {nazev}: rozdíl nejhorší × nejlepší decil = {d.iloc[-1]-d.iloc[0]:.0%} b.", loc="left")
osy[0].set_ylabel("Podíl, kdo koupil znovu"); osy[0].set_ylim(0,1.05)
fig.tight_layout()
fig.savefig("outputs/v2_25_cviceni3_decily_A_C.png", dpi=120)
plt.show()

