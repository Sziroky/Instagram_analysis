import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv

df = pd.read_csv('D:\Kurs_SDA\Python_SDA\Final_Project\IG_small_buisness_proceed.csv')

'''
Struktura Nodes do GEPHI
ID      LABEL       INT DATA -- like no. of comments, likes, engagment etc. 
'''
df.drop(['Unnamed: 0.1', 'Unnamed: 0','media_type','date','hashtags','description','Tagged_profile','emoticons','post_url','post_id'],axis=1,inplace=True)

# konwersja na format czasowy
df['time_added'] = pd.to_datetime(df['time_added'], format='%H:%M:%S')

'''Feature engineering -- wartości średnie polubień,komentarzy i zaangażowania'''
# wyliczenie średnich
likes= dict(df.groupby('username')['likes'].mean())
com= dict(df.groupby('username')['comments'].mean())
eng= dict(df.groupby('username')['engagment'].mean())
# stworzenie kluczy
df['mean_likes'] = df['username']
df['mean_comments'] = df['username']
df['mean_engagment'] = df['username']
# mapowanie średnich według użytkownika
df['mean_likes'] = df['mean_likes'].map(likes)
df['mean_comments'] = df['mean_comments'].map(com)
df['mean_engagment'] = df['mean_engagment'].map(eng)

def lenght_of_description(text):
    return len(text)

df['descriptionlenght'] = df['Tokenized'].apply(lenght_of_description)

'''EDA'''

fig,ax=plt.subplots(figsize = (16,10))
corr = df.select_dtypes(exclude=['object']).corr()
sns.heatmap(corr,cmap="YlGnBu",fmt = '.1%', annot=True)
plt.title("Korelacje różnych zmiennych dla 42 profili w serwisie Instagram\n Tematyka konta: small biznes lokalizacja: Polska\n")
fig.savefig('Corelation.png')

fig,ax = plt.subplots(figsize=(16,10))
ax.bar(df['time_added'],df['likes'],color='r',label = "Polubienia")
ax.bar(df['time_added'],df['comments'],color='b',label = "Komentarze")
ax.set_xlabel('Time(Hour/Minute/Seconds)')
ax.set_ylabel('Number')
ax.set_title("Ilość komentarzy i polubień w zależności od godziny")
ax.legend()
ax.xaxis.set_tick_params(rotation=90)
fig.savefig('test.png')

report = sv.analyze(df)
report.show_html('sweetviz_report.html')