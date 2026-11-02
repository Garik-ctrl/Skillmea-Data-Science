import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import alpha

#nějaká úprava

fig, ax = plt.subplots() # figure - "plátno", axes - samotný graf
x=[0,1,2,3] #zdrojová data osa x
y=[0,1,2,3] #zdrojová data osa y
"""
ax.grid(
    linestyle="--", #styl čáry
    linewidth=2, # tloušťka nad 1 se zvětšuje, jinak zmenšuje
    color="#42DB8B", # barva
    alpha=0.1 # průhlednost
)
"""
ax.grid(
    axis="x",
    linestyle="--", #styl čáry
    linewidth=2, # tloušťka nad 1 se zvětšuje, jinak zmenšuje
    color="#42DB8B", # barva
    alpha=0.8 # průhlednost
)
ax.grid(
    axis="y",
    linestyle="--", #styl čáry
    linewidth=2, # tloušťka nad 1 se zvětšuje, jinak zmenšuje
    color="#2F756E", # barva
    alpha=0.9 # průhlednost
)

ax.plot(x, y)
#plt.show()

df = pd.read_csv(r"C:\Users\mistr\Downloads\data\data\weather.csv", sep=",",index_col='date')
#index_col - osa x

pocet_bodu=15
#df1=df[0:10]
fig, ax = plt.subplots(figsize=(18,3))
ax.plot(df.index[:pocet_bodu],df['temp_max'][:pocet_bodu],label='Max',color='red',marker='o')
ax.plot(df.index[:pocet_bodu], df['temp_min'][:pocet_bodu],
        color='blue',
        marker='o',
        linestyle='None',
        label='Min')

ax.set_title('Teploty ve meste')
ax.set_ylabel('Teplota ve stupnich')
ax.set_xlabel('Datum')



plt.xticks(rotation=45)
plt.legend()
plt.show()
#----------------------------------------------------
from matplotlib.ticker import MaxNLocator
fig, ax = plt.subplots(figsize=(12,4))
ax.xaxis.set_major_locator(MaxNLocator(10))
ax.plot(df.index, df['temp_max'])
ax.plot(df.index, df['temp_min'])

plt.show()
#----------------------------------------------------
pocet_bodu=20
#df1=df[0:10]
fig, ax = plt.subplots(2,1, figsize=(6,1))
ax[0].plot(df.index[:pocet_bodu],df['temp_max'][:pocet_bodu],label='Max',color='red',marker='o')
ax[1].plot(df.index[:pocet_bodu], df['temp_min'][:pocet_bodu],
        color='blue',
        marker='o',
        linestyle='None',
        label='Min')

ax[0].set_title('Teploty ve meste')
ax[0].set_ylabel('Teplota ve stupnich')
ax[0].set_xlabel('Datum')

ax[1].set_title('Teploty na vesnici')
ax[1].set_ylabel('Zimní teplota')
ax[1].set_xlabel('Datum')
ax[0].grid()
ax[0].xaxis.set_tick_params(rotation=45)
ax[1].xaxis.set_tick_params(rotation=-45)
plt.show()
ax[0].annotate(
    "Covid",
    xy=(df.index[18],df['temp_max'].iloc[5]), # kde je ta šipka
    xytext=(df.index[10],df['temp_max'].iloc[5]+3), # kde bude text
    arrowprops=dict(arrowstyle='->', color='black', lw=0.5)
)
plt.show()
#--------------------------------------------------------------------------
population = pd.read_csv(r"C:\Users\mistr\Downloads\data\data\population.csv", sep=",", index_col='City')
#index_col - osa x
print(population.head())
print(population.info())

fig, ax = plt.subplots(figsize=(12,4))
ax.bar(population.index, population['Value'])
ax.set_title('Populace v jednotlivých mestech')
ax.set_ylabel('Populace')
ax.set_xlabel('City')
plt.xticks(rotation=90)
plt.show()

muzi=population[population["Sex"]=="Male"]
zeny=population[population["Sex"]=="Female"]
dohromady=population[population["Sex"]=="Both Sexes"]
print(muzi.head())

fig, ax = plt.subplots(2,1,figsize=(12,4))
ax[0].bar(muzi.index, muzi['Value'])
ax[0].set_title('Populace v jednotlivých mestech(muzi)')
ax[0].set_ylabel('Populace')
ax[0].set_xlabel('City')

ax[1].bar(zeny.index, zeny['Value'])
ax[1].set_title('Populace v jednotlivých mestech(zeny)')
ax[1].set_ylabel('Populace')
ax[1].set_xlabel('City')
plt.xticks(rotation=90)
plt.show()
#----------------------------------------------
muzi_value=population[population["Sex"]=="Male"]["Value"]
zeny_value=population[population["Sex"]=="Female"]["Value"]


fig, ax= plt.subplots(figsize=(12,4))
ax.hist(muzi_value, bins=9,color='blue', alpha=0.5,label="muzi")
ax.hist(zeny_value, bins=9,color='pink', alpha=0.5,label="zeny")
plt.legend()
plt.show()

#-----------------------------
#boxplot
fig, ax= plt.subplots(figsize=(12,4))
ax.boxplot(muzi_value, zeny_value)
ax.set_xticklabels(["muzi", "zeny"])

plt.show()


