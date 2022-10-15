import pandas as pd
import matplotlib.pyplot as plt


def problem_1():
    years = [i for i in range(2011, 2021)]
    durations = [103, 101, 99, 100, 100, 95, 95, 96, 93, 90]

    mydict = dict()
    for y, d in zip(years, durations):
        mydict[y] = d
    return mydict

def problem_2(mydict):
    df = pd.DataFrame({
        'year': mydict.keys(),
        'duration': mydict.values()
    })
    return df

def problem_3(df, plot = True):
    plt.figure(figsize=(12, 9))
    plt.title("Movie Durations 2011-2020")
    plt.plot(df['year'], df['duration'])
    
    if plot:
        plt.show()

    plt.close()
    
    return df['year'], df['duration']

def problem_4(file_name):
    df = pd.read_csv(file_name)
    return df

def problem_5(df):
    df = df[df['type'] == 'Movie']
    df = df[['title', 'country', 'genre', 'release_year', 'duration']]
    df = df[df['duration'].notna()]
    df['duration'] = df['duration'].apply(lambda x: int(x.replace('min', '')))
    return df

def problem_6(df, plot = True):
    plt.figure(figsize=(12, 9))
    plt.title("Movie Duration by Year of Release")
    plt.scatter(df['release_year'], df['duration'])

    if plot:
        plt.show()

    plt.close()
    
    return df['release_year'], df['duration']


def problem_7(df):
    df = df[df['duration'] < 60]
    return df.head(20)

def problem_8(df):
    colors = list()
    for genre in df['genre']:
        if genre == 'Children & Family Movies':
            colors.append('red')
        elif genre == 'Documentaries':
            colors.append('blue')
        elif genre == 'Stand-Up Comedy':
            colors.append('green')
        else:
            colors.append('black')
    return colors

def problem_9(df, colors, plot = True):
    # Use the following snippet of code in your function for visualization purposes only.
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12,8))
    plt.title("Movie duration by year of release")
    plt.xlabel("Release year")
    plt.ylabel("Duration (min)")
    plt.scatter(df['release_year'], df['duration'], color=colors)

    if plot:
        plt.show()

    plt.close()

    return df['release_year'], df['duration'], colors

def problem_10():
    ans = 'Yes'
    return ans
