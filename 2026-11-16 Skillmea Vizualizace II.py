import pandas as pd                                          # pandas = tabulky (DataFrame)
import numpy as np                                           # numpy = matematika nad celými sloupci
import matplotlib.pyplot as plt                              # matplotlib = základ všech grafů v Pythonu
import matplotlib.colors as mcolors                          # nástroje pro barvy (použijeme u mapy krajů)
import seaborn as sns                                        # seaborn = nadstavba nad matplotlib: heatmapy, boxploty, violiny jedním řádkem
import os                                                    # práce se složkami
from sklearn.model_selection import train_test_split         # rozdělení na train a test
from sklearn.linear_model import LogisticRegression          # model ano/ne
from sklearn.metrics import roc_auc_score                    # AUC = jak dobře model řadí zákazníky
from sklearn.metrics import roc_curve                        # NOVĚ: body ROC křivky (fpr, tpr) pro kreslení
from sklearn.metrics import confusion_matrix                 # NOVĚ: matice záměn (kolik tipů trefilo, kolik minulo)

os.makedirs("outputs", exist_ok=True)
ODRA="#2a78d6"
SVETLA_MODRA="#a9c8f0"
ORANZOVA="#eb6834"
SEDA="#52514e"
SVETLE_SEDA="#c9c8c1"

BARVY_SEGMENT={"Standard":MODRA, "Premium":ORANZOVA, "Firemní":"#2f8f6b"}
PORADI_SEGMENT=["Standard","Premium","Firemní"]

plt.rcParams.update({"axes.spines.top":False, "axes.spines.right":False,
                     "axes.grid":True, "grid.color":"#e6e5e1", "grid.linewidth":0.8,
                     "axes.axisbelow":True,
                     "axes.edgecolor":"#c3c2b7", "axes.labelcolor":SEDA,
                     "xtick.color":SEDA, "ytick.color":SEDA, "font.size":10,
                     "axes.titleweight":"bold", "axes.titlesize":11})

objednavky=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\objednavky2.csv")
produkty=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\produkty2.csv")
zakaznici=pd.read_csv(r"C:\Users\mistr\Downloads\Vlnka Shop - cista verze\data\zakaznici2.csv")

objednavky["order_date"]=pd.to_datetime(objednavky["order_date"], format="%Y-%m-%d %H:%M")

op=objednavky.merge(produkty, on="product_id", how="left", validate="m:1")
assert len(op)==len(objednavky)
op["trzba_bez_dph"]=op["quantity"]*op["cena_za_kus_bez_dph"]*(1-op["discount_pct"]/100)   # tržba řádku

agg_tabulka=(op
             .groupby("order_id")
             .agg(order_date=("order_date","first"),
                  customer_id=("customer_id","first"),
                  status=("status","first"),
                  channel=("channel","first"),
                  dopravce=("dopravce","first"),
                  delivery_days=("delivery_days","first"),
                  trzba_bez_dph=("trzba_bez_dph","sum"))
             .reset_index()
             .sort_values(["order_date","order_id"]))
print("Objednávek:", len(agg_tabulka))


dok=agg_tabulka[agg_tabulka["status"]=="dokonceno"].copy()
dok=dok.merge(zakaznici[["customer_id","segment","kraj","zdroj"]], on="customer_id", how="left", validate="m:1")
assert len(dok)==23940
dok["rok"]=dok["order_date"].dt.year
dok["mesic"]=dok["order_date"].dt.month
dok["rok_mesic"]=dok["order_date"].dt.to_period("M")
dok["den_v_tydnu"]=dok["order_date"].dt.dayofweek
dok["hodina"]=dok["order_date"].dt.hour


hranice=pd.Timestamp("2026-06-30 23:59")
konec_okna=hranice+pd.Timedelta(days=123)
obchody_historie=dok[dok["order_date"]<=hranice]
obchody_budoucnost=dok[(dok["order_date"]>hranice)&(dok["order_date"]<=konec_okna)]

zakaznik=pd.DataFrame(index=sorted(obchody_historie["customer_id"].unique()))
zakaznik.index.name="customer_id"
zakaznik["koupi_znovu"]=zakaznik.index.isin(obchody_budoucnost["customer_id"]).astype(int)
print("Zákazníků:", len(zakaznik), "| podíl koupí znovu:", round(zakaznik["koupi_znovu"].mean(),3))

nakupy=(obchody_historie.groupby("customer_id")
        .agg(posledni_nakup=("order_date","max"), frekvence=("order_id","count"), monetary=("trzba_bez_dph","sum")))
zakaznik=zakaznik.join(nakupy)
zakaznik["recency_dny"]=(hranice-zakaznik["posledni_nakup"]).dt.days
zakaznik["AVG_hodnota_objednavky"]=zakaznik["monetary"]/zakaznik["frekvence"]
zakaznik["monetary_log"]=np.log1p(zakaznik["monetary"])

h=obchody_historie.sort_values(["customer_id","order_date"]).copy()
h["interval"]=h.groupby("customer_id")["order_date"].diff().dt.days
zakaznik["interval_prumer"]=h.groupby("customer_id")["interval"].mean()

posledni90=obchody_historie[obchody_historie["order_date"]>hranice-pd.Timedelta(days=90)]
zakaznik["objednavek_90d"]=posledni90.groupby("customer_id").size().reindex(zakaznik.index, fill_value=0)

vsechny=agg_tabulka[agg_tabulka["order_date"]<=hranice]
zakaznik["podil_vracenych"]=(vsechny["status"]!="dokonceno").groupby(vsechny["customer_id"]).mean().reindex(zakaznik.index)   # podíl vrácených/zrušených

segment=zakaznici.set_index("customer_id")["segment"].reindex(zakaznik.index)
zakaznik["segment_premium"]=(segment=="Premium").astype(int)
zakaznik["segment_firemni"]=(segment=="Firemní").astype(int)

y=zakaznik["koupi_znovu"]
X=zakaznik.drop(columns=["koupi_znovu","posledni_nakup","monetary"])
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
median=X_train.median()
X_train=X_train.fillna(median)
X_test=X_test.fillna(median)
prumer=X_train.mean()
odchylka=X_train.std()
X_train_s=(X_train-prumer)/odchylka
X_test_s=(X_test-prumer)/odchylka

model=LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
p_test=model.predict_proba(X_test_s)[:,1]
print("AUC modelu:", round(roc_auc_score(y_test, p_test),3))

X_vse=X.fillna(median)
zakaznik["pravdepodobnost"]=model.predict_proba((X_vse-prumer)/odchylka)[:,1]
zakaznik["riziko"]=pd.cut(zakaznik["pravdepodobnost"], bins=[0,0.4,0.7,1], labels=["vysoké riziko","střední","věrný"])
zakaznik["segment"]=segment
zakaznik["kraj"]=zakaznici.set_index("customer_id")["kraj"].reindex(zakaznik.index)
zakaznik["zdroj"]=zakaznici.set_index("customer_id")["zdroj"].reindex(zakaznik.index)
