import pandas as pd


df = pd.read_csv(
    'koryukova.tsv', sep=';', skiprows=39, 
    usecols=[1, 2, 3, 4, 5], header=3,
    names=['Name', 'Freq', 'Epoch', 'Sizecore', 'e_Sizecore'],
)
df.Epoch = pd.to_datetime(df.Epoch)
print(df.Sizecore[df.Sizecore > 3].shape)
print(df[df.Sizecore > 5].reset_index(drop=True))