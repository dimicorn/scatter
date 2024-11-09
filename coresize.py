import pandas as pd


df = pd.read_csv(
    'koryukova.tsv', sep=';', skiprows=39, 
    usecols=[1, 2, 3, 4, 5, 6, 7], header=3,
    names=['Name', 'Freq', 'Epoch', 'Sizecore', 'e_Sizecore', 'GLAT', 'GLON'],
)
df.Epoch = pd.to_datetime(df.Epoch)
# print(df.Sizecore[df.Sizecore > 3].shape)
df = df[df.Sizecore > 5]
df = df.sort_values(by=['Sizecore'], ascending=False).reset_index(drop=True)
df.head(20).to_csv('top_20_2.csv', index=False)