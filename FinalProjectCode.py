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
