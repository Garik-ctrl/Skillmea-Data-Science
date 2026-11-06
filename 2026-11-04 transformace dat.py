import pandas as pd

objednavky=pd.read_csv(r"C:\Users\mistr\Downloads\skillmea_ekce2\data\objednavky.csv")
produkty=pd.read_csv(r"C:\Users\mistr\Downloads\skillmea_ekce2\data\produkty.csv")
zakaznici=pd.read_csv(r"C:\Users\mistr\Downloads\skillmea_ekce2\data\zakaznici.csv")
print(objednavky.isna().sum())

print("objednavky:")
objednavky["order_date"]=pd.to_datetime(objednavky["order_date"], format="%Y-%m-%d %H:%M")
print(objednavky["order_date"].head())
print(objednavky.info())


print("zakaznici:")
print(zakaznici.head())
print(zakaznici.info())
zakaznici["registrace"]=pd.to_datetime(zakaznici["registrace"], format="%Y-%m-%d")

print("produkty:")
print(produkty.head())
print(produkty.info())

#mini příklad

obj_mini=pd.DataFrame({"obj":["A","B","C","D","E"], "prod":["P1","P2","P2", "P3","P9"]})
kat_mini=pd.DataFrame({"prod":["P1","P2","P3","P4"], "nazev":["mys","klavesnice","notebook","monitor"]})
print(obj_mini)
print(kat_mini)

#merge
spojena_mini_tabulka=obj_mini.merge(kat_mini, on="prod", how="outer")
print(spojena_mini_tabulka)
#---------------------------------------------------------------------------------------------------------
objednavky["hruba_trzba"]=objednavky["quantity"]*objednavky["cena_za_kus_bez_dph"]
print(objednavky.info())

print((objednavky["quantity"]*objednavky["cena_za_kus_bez_dph"]).sum())
print(objednavky["hruba_trzba"].sum())

#---------------------------------------------------------------------------------------------------------
pocet_radku_pred=len(objednavky)
print(pocet_radku_pred)
spojena_tabule_obj_prod=objednavky.merge(produkty, on="product_id", how="left")
print(spojena_tabule_obj_prod.info())

megatabulka=spojena_tabule_obj_prod.merge(zakaznici, on="customer_id", how="left")
print(megatabulka.info())

megatabulka["customer_id"]=megatabulka["customer_id"].fillna("Nezaregistrovan")
megatabulka["rok_narozeni"]=megatabulka["rok_narozeni"].fillna(2100)
megatabulka["rok_narozeni"]=megatabulka["rok_narozeni"].fillna(megatabulka["rok_narozeni"].median())
#---------------------------------------------------------------------------------------------------------
#dt.year - vyextrahuje z toho rok
rok2025=objednavky[objednavky["order_date"].dt.year==2025]
rok2026=objednavky[objednavky["order_date"].dt.year==2026]

print(objednavky.info())
print(rok2025["order_date"])
#concat
spojeno_2025_2026=pd.concat([rok2025,rok2026],ignore_index=True)
print("pocet zaznamu rok2025 ", len(rok2025), " pocet zaznamu rok2026 ", len(rok2026), " pocet zaznamu dohromady ", len(spojeno_2025_2026))
#---------------------------------------------------------------------------------------------------------
print(objednavky.isna().sum())
objednavky["chybejici_sleva"]=objednavky["discount_pct"]
objednavky["chybejici_sleva"]=objednavky["chybejici_sleva"].fillna(0)
print(objednavky["chybejici_sleva"])
#---------------------------------------------------------------------------------------------------------
print(megatabulka.info())
print(megatabulka[["quantity","cena_za_kus_bez_dph","naklad"]])
#hrubý zisk = (cena_bez_DPH-naklad)*quantity
megatabulka["hruby_zisk"]=(megatabulka["cena_za_kus_bez_dph"]-megatabulka["naklad"])*megatabulka["quantity"]
print(megatabulka[["quantity","cena_za_kus_bez_dph","naklad","hruby_zisk"]])
print(megatabulka["hruby_zisk"].sum())
megatabulka["cena_za_kus_s_dph"]=megatabulka["cena_za_kus_bez_dph"]*1.21
print(megatabulka[["cena_za_kus_bez_dph","cena_za_kus_s_dph"]])

print(megatabulka["order_id"])
megatabulka["order_id"]=megatabulka["order_id"].str.replace("OBJ-","").astype(int)
print(megatabulka.info())
#datumové operace
megatabulka["rok_objednavky"]=megatabulka["order_date"].dt.year #2025,2026
megatabulka["mesic_objednavky"]=megatabulka["order_date"].dt.month #1-12
megatabulka["rok_mesic_objednavky"]=megatabulka["order_date"].dt.to_period("M")
megatabulka["den_v_tydnu_objednavky"]=megatabulka["order_date"].dt.day_of_week+1 #0 - pondělí 6 - neděle
megatabulka["den_v_tydnu_nazev_objednavky"]=(megatabulka["order_date"]
                                             .dt.day_of_week
                                             .map({0:"Pondeli",1:"Utery",2:"Streda", 3:"Ctvrtek", 4:"Patek", 5: "Sobota", 6:"Nedele"}))
print(megatabulka["den_v_tydnu_nazev_objednavky"])
print(megatabulka[["order_date","registrace"]])

megatabulka["rozdil_v_datech"]=(megatabulka["order_date"]-megatabulka["registrace"]).dt.days
print(megatabulka["rozdil_v_datech"])

# tržby z dokončených objednavek
print(megatabulka.info())
print(megatabulka[megatabulka["status"]=="dokonceno"]["hruby_zisk"].sum())

dokoncene_objednavky_mega=megatabulka[megatabulka["status"]=="dokonceno"]
#tržby z dokončených objednávek v jednotlivých měsících

trzby_dle_mesice=dokoncene_objednavky_mega.groupby(["rok_objednavky", "mesic_objednavky"])["hruby_zisk"].sum()
print(trzby_dle_mesice)

print(megatabulka["hruby_zisk"].describe())

import numpy as np
megatabulka["velikost_objednavky"]=np.where(megatabulka["hruby_zisk"]>750,"velka","bezna")
print(megatabulka[["hruby_zisk","velikost_objednavky"]])

# trzby dle objednavek a roku+mesic

trzby_dle_mesice_obj=dokoncene_objednavky_mega.groupby(["order_id", "rok_mesic_objednavky"])["hruby_zisk"].sum()
print(trzby_dle_mesice_obj)
print(trzby_dle_mesice_obj.info())

pomocna_tabulka=zakaznici.groupby("segment")["registrace"].count()
print(pomocna_tabulka)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16,9))
ax.bar(pomocna_tabulka.index, pomocna_tabulka.values)
plt.show()
