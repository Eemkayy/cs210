import pandas as pd
from matplotlib import pyplot as plt


def genre_trend_per_year(df):
    genre_count_per_year = df.groupby(['Year', 'Genre']).size().unstack(fill_value=0)
    genre_count_per_year['Total'] = genre_count_per_year.sum(axis=1)

    genre_count_per_year.drop(columns='Total', inplace=True)

    genre_count_per_year.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Genre Trends Over The Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Books')
    plt.xticks(rotation=45)
    plt.legend(title='Genre')
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()