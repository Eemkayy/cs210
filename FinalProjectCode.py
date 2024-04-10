import matplotlib.ticker as ticker
import numpy as np
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

def repeat_best_sellers(df):
    repeat_bestsellers = df.groupby(['Name', 'Author']).filter(lambda x: len(x) > 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    for genre, group in repeat_bestsellers.groupby('Genre'):
        ax.boxplot(group['Reviews'], positions=[1 if genre == 'Fiction' else 2], widths=0.6, labels=[genre])
    ax.set_title('Distribution of Reviews for Repeat Bestsellers by Genre')
    ax.set_ylabel('Reviews')
    plt.show()

def correlation_cost_reviews_rating(df):
    correlation_matrix = df[['User Rating', 'Reviews', 'Price']].corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_yticks(np.arange(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns)
    ax.set_yticklabels(correlation_matrix.columns)

    plt.title('Correlation Matrix: Rating, Reviews, and Price')
    plt.show()

def genre_performance_comparison (df):
    genre_means = df.groupby('Genre').agg({'User Rating': 'mean', 'Reviews': 'mean'}).reset_index()
    genres = genre_means['Genre']
    ratings_mean = genre_means['User Rating']
    reviews_mean = genre_means['Reviews'].astype(float)
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Genre')
    ax1.set_ylabel('Average User Rating', color=color)
    ax1.bar(genres, ratings_mean, color=color, alpha=0.6, label='User Rating')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yticks(np.arange(0, 5.5, 0.5))  
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Average Reviews (in decimals)', color=color)
    ax2.plot(genres, reviews_mean, color=color, marker='o', linestyle='-', linewidth=2, markersize=12, label='Reviews')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Average User Rating and Reviews by Genre')
    fig.tight_layout()  
    plt.show()

def analysis_price_cultural_relevance(df):

    df['Price Range'] = pd.cut(df['Price'], bins=[0, 10, 20, 30, np.inf], labels=['$0-10', '$10-20', '$20-30', '$30+'])

    fig, ax = plt.subplots(figsize=(10, 6))
    price_ranges = df['Price Range'].unique().dropna()
    reviews = [df[df['Price Range'] == price]['Reviews'] for price in price_ranges]
    ax.boxplot(reviews, labels=price_ranges)
    ax.set_title('Reviews Distribution by Price Range')
    ax.set_ylabel('Reviews')
    plt.show()


def plot_retention_rate(df):
    grouped = df.groupby('Year')
    retention_rates = []

    for year, group in grouped:
        next_year = year + 1
        if next_year in grouped.groups:
            current_books = set(zip(group['Name'], group['Author']))
            next_year_books = set(zip(grouped.get_group(next_year)['Name'], grouped.get_group(next_year)['Author']))
            retention_rate = len(current_books.intersection(next_year_books)) / len(current_books)
            retention_rates.append((year, next_year, retention_rate))

    years = [f"{year} to {next_year}" for year, next_year, _ in retention_rates]
    rates = [rate for _, _, rate in retention_rates]

    plt.figure(figsize=(10, 6))
    plt.bar(years, rates, color='skyblue')
    plt.title('Retention Rate of Books per Year')
    plt.xlabel('Year Range')
    plt.ylabel('Retention Rate')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_retention_rate_by_criteria(df, group_by='Genre', group_by_title='Genre', cut_range = [0, 10, 20, 30, np.inf], cut_labels=['$0-10', '$10-20', '$20-30', '$30+'], to_cut = False):
    if to_cut:
        df[f'{group_by} Range'] = pd.cut(df[group_by], bins=cut_range, labels=cut_labels)

        group_by = group_by + " Range"
    
    grouped_by_year = df.groupby('Year')
    years = sorted(df['Year'].unique())
    crit = df[group_by].unique()
    
    retention_data = {year: {field: None for field in crit} for year in years[:-1]} 
    
    for year, group in grouped_by_year:
        next_year = year + 1
        if next_year in years:
            for field in crit:
                current_books = set(zip(group[group[group_by] == field]['Name'], group[group[group_by] == field]['Author']))
                if next_year in grouped_by_year.groups:
                    next_year_books = set(zip(grouped_by_year.get_group(next_year)[grouped_by_year.get_group(next_year)[group_by] == field]['Name'], 
                                              grouped_by_year.get_group(next_year)[grouped_by_year.get_group(next_year)[group_by] == field]['Author']))
                    if current_books:
                        retention_rate = len(current_books.intersection(next_year_books)) / len(current_books)
                    else:
                        retention_rate = 0 
                    retention_data[year][field] = retention_rate
    fig, ax = plt.subplots(figsize=(12, 8))
    width = 0.75 / len(crit)
    for i, field in enumerate(crit):
        rates = [retention_data[year][field] for year in years[:-1]]
        ax.bar(np.arange(len(years)-1) + i*width, rates, width, label=field)

    ax.set_xticks(np.arange(len(years)-1) + width / 2)
    ax.set_xticklabels(years[:-1])
    ax.set_ylabel('Retention Rate')
    ax.set_xlabel('Year')
    ax.set_title(f'Retention Rates by {group_by_title} per Year')
    ax.legend(title=f'{group_by_title}')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def change_over_time(df):
    books_per_year_genre = df.groupby(['Year', 'Genre']).size().unstack(fill_value=0)

    yoy_changes = books_per_year_genre.diff().fillna(0)
    yoy_changes.plot(kind='bar', figsize=(14, 8))
    plt.title('Year-over-Year Change in Number of Books by Genre')
    plt.xlabel('Year')
    plt.ylabel('Change in Number of Books')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.legend(title='Genre')
    plt.tight_layout()
    plt.show()


def trend_nf(df):
    non_fiction_data = df[df['Genre'] == 'Non Fiction']

    books_per_year = non_fiction_data.groupby('Year')['Name'].count()
    average_price_per_year = non_fiction_data.groupby('Year')['Price'].mean()


    fig, ax1 = plt.subplots()


    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Top 50 Books', color=color)
    ax1.scatter(books_per_year.index, books_per_year.values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)


    slope, intercept = np.polyfit(books_per_year.index, books_per_year.values, 1)
    line = slope * books_per_year.index + intercept
    ax1.plot(books_per_year.index, line, color=color, label='Popularity Trend')


    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Average Price ($)', color=color)
    ax2.plot(average_price_per_year.index, average_price_per_year.values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    slope, intercept = np.polyfit(average_price_per_year.index, average_price_per_year.values, 1)
    line = slope * average_price_per_year.index + intercept
    ax2.plot(average_price_per_year.index, line, color=color, linestyle='--', label='Price Trend')

    plt.title('Popularity and Price Trend of Non Fiction Books')
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)

    plt.show()


def trend_f(df):
    non_fiction_data = df[df['Genre'] == 'Fiction']

    books_per_year = non_fiction_data.groupby('Year')['Name'].count()
    average_price_per_year = non_fiction_data.groupby('Year')['Price'].mean()

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Top 50 Books', color=color)
    ax1.scatter(books_per_year.index, books_per_year.values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    slope, intercept = np.polyfit(books_per_year.index, books_per_year.values, 1)
    line = slope * books_per_year.index + intercept
    ax1.plot(books_per_year.index, line, color=color, label='Popularity Trend')
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Price ($)', color=color)
    ax2.plot(average_price_per_year.index, average_price_per_year.values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    slope, intercept = np.polyfit(average_price_per_year.index, average_price_per_year.values, 1)
    line = slope * average_price_per_year.index + intercept
    ax2.plot(average_price_per_year.index, line, color=color, linestyle='--', label='Price Trend')
    plt.title('Popularity and Price Trend of Fiction Books')
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)

    plt.show()  