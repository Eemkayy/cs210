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

def which_books_top_50_but_not_last_year(df, current_year, genre):
    current_year_df = df[(df['Year'] == current_year) & (df['Genre'] == genre)]
    previous_year_df = df[(df['Year'] == current_year - 1) & (df['Genre'] == genre)]
    new_books_this_year = current_year_df[~current_year_df['Name'].isin(previous_year_df['Name'])]
    
    return new_books_this_year


def average_price_per_genre_between_years(df, start_year, end_year):
    filtered_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
    avg_price_per_genre_year = filtered_df.groupby(['Year', 'Genre'])['Price'].mean().unstack()
    
    return avg_price_per_genre_year