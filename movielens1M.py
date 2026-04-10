import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns

path = '/Users/luanabreno/Downloads/ml-1m/'

unames = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
users = pd.read_csv(path + 'users.dat', sep='::', engine='python',
                    header=None, names=unames, encoding='latin-1')

rnames = ['user_id', 'movies_id', 'rating', 'timestamp']
ratings = pd.read_csv(path + 'ratings.dat', sep='::', engine='python',
                      header=None, names=rnames, encoding='latin-1')

mnames = ['movies_id', 'title', 'genres']
movies = pd.read_csv(path + 'movies.dat', sep='::', engine='python',
                     header=None, names=mnames, encoding='latin-1')

    # Checking the data
print(ratings.head())
print(movies.head())
print(users.head())

    # Merging tables
data = pd.merge(pd.merge(ratings, users), movies)
print(data.head())

    # Histogram of ratings (1-5)
sns.set_style("whitegrid")
sns.histplot(data['rating'], bins=5, color='lightgreen', edgecolor='black', discrete=True)
plt.xlabel('Rating')
plt.ylabel('Count')
#plt.show()

    # Barplot top 10 movies by rating
top_10 = (data.groupby('title')['rating']
          .agg(['count', 'mean'])
          .nlargest(10, 'count')
          .sort_values('mean', ascending=False))

print(top_10)
plt.figure(figsize=(12,5))
sns.barplot(x=top_10['mean'], y=top_10.index, color='lightgreen', edgecolor='black', orient='h')
plt.xlabel('Average Rating')
plt.ylabel('Movie')
plt.title('Top 10 Movies')
plt.xlim(top_10['mean'].min() - 0.1, top_10['mean'].max() + 0.1)
plt.tight_layout()
#plt.show()

    #  Creating a model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, mean_absolute_error

data['genres_list'] = data['genres'].str.split('|')

train, test = train_test_split(data, test_size=0.3, random_state=42)

mlb = MultiLabelBinarizer()
    # Transforming the data
mlb.fit(train['genres_list'])

genre_train = mlb.transform(train['genres_list'])
genre_test = mlb.transform(test['genres_list'])

user_mean_train = train.groupby('user_id')['rating'].mean()
movie_mean_train = train.groupby('movies_id')['rating'].mean()

train['user_mean'] = train['user_id'].map(user_mean_train)
test['user_mean'] = test['user_id'].map(user_mean_train)

train['movie_mean'] = train['movies_id'].map(movie_mean_train)
test['movie_mean'] = test['movies_id'].map(movie_mean_train)

    # Filtering NaN
global_mean = train['rating'].mean()

train.fillna({'user_mean': global_mean}, inplace=True)
test.fillna({'user_mean': global_mean}, inplace=True)

train.fillna({'movie_mean': global_mean}, inplace=True)
test.fillna({'movie_mean': global_mean}, inplace=True)

genre_train_df = pd.DataFrame(genre_train, columns=mlb.classes_, index=train.index)
genre_test_df = pd.DataFrame(genre_test, columns=mlb.classes_, index=test.index)

X_train = pd.concat([train[['user_mean', 'movie_mean']], genre_train_df], axis=1)
X_test = pd.concat([test[['user_mean', 'movie_mean']], genre_test_df], axis=1)

y_train = train['rating']
y_test = test['rating']

models = {
    'linear': LinearRegression(),
    'rf': RandomForestRegressor(n_estimators=50, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    # Mean Absolute Error
    mae = mean_absolute_error(y_test, pred)
    print(f'{name}:\nRMSE = {rmse:.2f}, MAE = {mae:.2f}\n\n')

"""
Comparing models to predict rating from genres:
    - LinearRegression: simple baseline, assumes linear relationship. (RMSE= 0.93, MAE= 0.74)
    - RandomForest: captures genre interactions. (RMSE= 1.00, MAE= 0.78)
"""

# Recommending Similar Movies
moviemat = data.pivot_table(index = 'user_id', columns = 'title', values = 'rating')

# Top 1 movie is - Star Wars: Episode IV - A New Hope (1977) - according to our previous barplot...
#... let's try to recommend movies similar to this one.
starwars_user_ratings = moviemat['Star Wars: Episode IV - A New Hope (1977)']
# finding correlations
similar_to = moviemat.corrwith(starwars_user_ratings)
corr_starwars = similar_to.to_frame(name='Correlation')
corr_starwars.dropna(inplace=True)
corr_starwars['num of ratings']  = pd.DataFrame(data.groupby('title')['rating'].count())

# I'm gonna consider the correlation only for movies with more than 100 ratings
recomendations = (corr_starwars[corr_starwars['num of ratings'] > 100]
                  .sort_values('Correlation', ascending=False).head())

# - If you liked Star Wars... maybe you'll like these movies too!
print(recomendations)
# Result
"""
                                                    Correlation  num of ratings
title                                                                          
Star Wars: Episode IV - A New Hope (1977)              1.000000            2991
Star Wars: Episode V - The Empire Strikes Back ...     0.661552            2990
Star Wars: Episode VI - Return of the Jedi (1983)      0.574808            2883
Raiders of the Lost Ark (1981)                         0.421425            2514
Dracula (1958)                                         0.398710             102
"""


