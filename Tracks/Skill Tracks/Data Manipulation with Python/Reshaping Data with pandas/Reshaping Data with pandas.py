## The long and the wide

# Long format
# There are multiple records ...
# Each row represents one feature
# Needs a column ...

## Flipping players

# Set name as index
fifa_transpose = fifa_players.set_index('name')

# Print fifa_transpose
print(fifa_transpose)

# Filter the DataFrame to keep only height and weight columns
fifa_transpose = fifa_players.set_index('name')[['height', 'weight']]

# Print fifa_transpose
print(fifa_transpose)

# Change the DataFrame so rows become columns and vice versa
fifa_transpose = fifa_players.set_index('name')[['height', 'weight']].transpose()

# Print fifa_transpose
print(fifa_transpose)

## Dribbling the pivot method

# Pivot fifa_players to get overall scores indexed by name and identified by movement
fifa_overall = fifa_players.pivot(index='name', columns='movement', values='overall')

# Print fifa_overall
print(fifa_overall)

# Pivot fifa_players to get attacking scores indexed by name and identified by movement
fifa_attacking = fifa_players.pivot(index='name', columns='movement', values='attacking')

# Print fifa_attacking
print(fifa_attacking)

# Use the pivot method to get overall scores indexed by movement and identified by name
fifa_names = fifa_players.pivot(index='movement', columns='name', values='overall')

# Print fifa_names
print(fifa_names)

## Offensive or defensive player?

# Pivot fifa_players to get overall and attacking scores indexed by name and identified by movement
fifa_over_attack = fifa_players.pivot(index='name', 
                                      columns='movement', 
                                      values=['overall', 'attacking'])

# Print fifa_over_attack
print(fifa_over_attack)

# Use pivot method to get all the scores index by name and identified by movement
fifa_all = fifa_players.pivot(index='name', columns='movement')

# Print fifa_all
print(fifa_all)

## Replay that last move!

# Drop the fifth row to delete all repeated rows
fifa_no_rep = fifa_players.drop(4, axis=0)

# Print fifa_pivot
print(fifa_no_rep)  

# Pivot fifa players to get all scores by name and movement
fifa_pivot = fifa_no_rep.pivot(index='name', columns='movement') 

# Print fifa_pivot
print(fifa_pivot)  

## Reviewing the moves

# Discard the fifth row to delete all repeated rows
fifa_drop = fifa_players.drop(4, axis=0)

# Use pivot method to get all scores by name and movement
fifa_pivot = fifa_drop.pivot(index='name', columns='movement') 

# Print fifa_pivot
print(fifa_pivot)  

# Use pivot table to get all scores by name and movement
fifa_pivot_table = fifa_players.pivot_table(index='name', 
                                            columns='movement', 
                                            aggfunc='mean')
# Print fifa_pivot_table
print(fifa_pivot_table)

## Exploring the big match

# Use pivot table to display mean age of players by club and nationality
mean_age_fifa = fifa_players.pivot_table(index='nationality', 
                                         columns='club', 
                                         values='age', 
                                         aggfunc='mean')

# Print mean_age_fifa
print(mean_age_fifa)

# Use pivot table to display max height of any player by club and nationality
tall_players_fifa = fifa_players.pivot_table(index='nationality', 
                                             columns='club', 
                                             values='height', 
                                             aggfunc='max')

# Print tall_players_fifa
print(tall_players_fifa)

# Use pivot table to show the count of players by club and nationality and the total count
players_country = fifa_players.pivot_table(index='nationality', 
                                           columns='club', 
                                           values='name', 
                                           aggfunc='count', 
                                           margins=True)

# Print players_country
print(players_country)

## The tallest and the heaviest

# Define a pivot table to get the characteristic by nationality and club
fifa_mean = fifa_players.pivot_table(index=['nationality', 'club'], 
                                     columns='year')

# Print fifa_mean
print(fifa_mean)

# Set the appropriate argument to show the maximum values
fifa_mean = fifa_players.pivot_table(index=['nationality', 'club'], 
                                     columns='year', 
                                     aggfunc='max')

# Print fifa_mean
print(fifa_mean)

# Set the argument to get the maximum for each row and column
fifa_mean = fifa_players.pivot_table(index=['nationality', 'club'], 
                                     columns='year', 
                                     aggfunc='max', 
                                     margins=True)

# Print fifa_mean
print(fifa_mean)

## Gothic times

# Melt books_gothic using the title column as identifier
gothic_melted = books_gothic.melt(id_vars='title')

# Print gothic_melted
print(gothic_melted)

# Melt books_gothic using the title, authors, and publisher columns as identifier
gothic_melted_new = books_gothic.melt(id_vars=['title', 'authors', 'publisher'])

# Print gothic_melted_new
print(gothic_melted_new)

## Rating is not everything

# Melt publisher column using title and authors as identifiers
publisher_melted = books_gothic.melt(id_vars=['title', 'authors'], 
                                     value_vars='publisher')

# Print publisher_melted
print(publisher_melted)

# Melt rating and rating_count columns using the title as identifier
rating_melted = books_gothic.melt(id_vars='title', 
                                  value_vars=['rating', 'rating_count'])

# Print rating_melted
print(rating_melted)

# Melt rating and rating_count columns using title and authors as identifier
books_melted = books_gothic.melt(id_vars=['title', 'authors'], 
                                 value_vars=['rating', 'rating_count'])

# Print books_melted
print(books_melted)

## How is Frankenstein, Dorian Gray?

# Melt the rating and rating_count using title, authors and publisher as identifiers
books_ratings = books_gothic.melt(id_vars=['title', 'authors', 'publisher'], 
                                  value_vars=['rating', 'rating_count'])

# Print books_ratings
print(books_ratings)

# Assign the name feature to the new variable column
books_ratings = books_gothic.melt(id_vars=['title', 'authors', 'publisher'], 
                                  value_vars=['rating', 'rating_count'], 
                                  var_name='feature')

# Print books_ratings
print(books_ratings)

# Assign the name number to the new column containing the values
books_ratings = books_gothic.melt(id_vars=['title', 'authors', 'publisher'], 
                                  value_vars=['rating', 'rating_count'], 
                                  var_name='feature', 
                                  value_name='number')

# Print books_ratings
print(books_ratings)

## The golden age

# Reshape wide to long using title as index and version as new name, and extracting isbn prefix 
isbn_long = pd.wide_to_long(golden_age, 
                            stubnames='isbn', 
                            i='title', 
                            j='version')

# Print isbn_long
print(isbn_long)

# Reshape wide to long using title and authors as index and version as new name, and prefix as stubnames 
prefix_long = pd.wide_to_long(golden_age, 
                              stubnames='prefix', 
                              i=['title', 'authors'], 
                              j='version')

# Print prefix_long
print(prefix_long)

# Reshape wide to long using title and authors as index and version as new name, and prefix and isbn as wide column prefixes 
all_long = pd.wide_to_long(golden_age, 
                           stubnames=['isbn', 'prefix'], 
                           i=['title', 'authors'], 
                           j='version')

# Print all_long
print(all_long)

## Decrypting the code

# Reshape using author and title as index, code as new name and getting the prefix language and publisher
the_code_long = pd.wide_to_long(books_brown, 
                                stubnames=['language', 'publisher'], 
                                i=['author', 'title'], 
                                j='code')

# Print the_code_long
print(the_code_long)

# Specify underscore as the character that separates the variable names
the_code_long = pd.wide_to_long(books_brown, 
                                stubnames=['language', 'publisher'], 
                                i=['author', 'title'], 
                                j='code', 
                                sep='_')

# Print the_code_long
print(the_code_long)

# Specify that wide columns have a suffix containing words
the_code_long = pd.wide_to_long(books_brown, 
                                stubnames=['language', 'publisher'], 
                                i=['author', 'title'], 
                                j='code', 
                                sep='_', 
                                suffix='\w+')

# Print the_code_long
print(the_code_long)

## Time to read, Katniss!

# Modify books_hunger by resetting the index without dropping it
books_hunger.reset_index(drop=False, inplace=True)

# Reshape using title and language as index, feature as new name, publication and page as prefix separated by space and ending in a word
publication_features = pd.wide_to_long(books_hunger, 
                                       stubnames=['publication', 'page'], 
                                       i=['title', 'language'], 
                                       j='feature', 
                                       sep=' ', 
                                       suffix='\w+')

# Print publication_features
print(publication_features)

## Did you say dystopia?

# Split the index of books_dys by the hyphen 
books_dys.index = books_dys.index.str.split('-')

# Print books_dys
print(books_dys)

# Get the first element after splitting the index of books_dys
books_dys.index = books_dys.index.str.split('-').str.get(0)

# Print books_dys
print(books_dys)

# Split by the hyphen the index of books_dys
books_dys.index = books_dys.index.str.split('-').str.get(0)

# Concatenate the index with the list author_list separated by a hyphen
books_dys.index = books_dys.index.str.cat(author_list, sep='-')

# Print books_dys
print(books_dys)

## What's your rating, Harry?

# Concatenate the title and subtitle separated by "and" surrounded by spaces
hp_books['full_title'] = hp_books['title'].str.cat(hp_books['subtitle'], sep =" and ") 

# Print hp_books
print(hp_books)

# Concatenate the title and subtitle separated by "and" surrounded by spaces
hp_books['full_title'] = hp_books['title'].str.cat(hp_books['subtitle'], sep =" and ") 

# Split the authors into writer and illustrator columns
hp_books[['writer', 'illustrator']] = hp_books['authors'].str.split('/', expand=True) 

# Print hp_books
print(hp_books)

# Concatenate the title and subtitle separated by "and" surrounded by spaces
hp_books['full_title'] = hp_books['title'].str.cat(hp_books['subtitle'], sep =" and ") 

# Split the authors into writer and illustrator columns
hp_books[['writer', 'illustrator']] = hp_books['authors'].str.split('/', expand=True)

# Melt goodreads and amazon columns into a single column 
hp_melt = hp_books.melt(id_vars=['full_title', 'writer'], 
                        var_name='source', 
                        value_vars=['goodreads', 'amazon'], 
                        value_name='rating')

# Print hp_melt
print(hp_melt)

## Elementary, dear Watson!

# Split main_title by a colon and assign it to two columns named title and subtitle 
books_sh[['title', 'subtitle']] = books_sh['main_title'].str.split(':', expand=True)

# Print books_sh
print(books_sh)

# Split main_title by a colon and assign it to two columns named title and subtitle 
books_sh[['title', 'subtitle']] = books_sh['main_title'].str.split(':', expand=True)

# Split version by a space and assign the second element to the column named volume
books_sh['volume'] = books_sh['version'].str.split(' ').str.get(1)

# Print books_sh
print(books_sh)

# Split main_title by a colon and assign it to two columns named title and subtitle 
books_sh[['title', 'subtitle']] = books_sh['main_title'].str.split(':', expand=True)

# Split version by a space and assign the second element to the column named volume
books_sh['volume'] = books_sh['version'].str.split(' ').str.get(1)

# Drop the main_title and version columns modifying books_sh
books_sh.drop(['main_title', 'version'], axis=1, inplace=True)

# Print books_sh
print(books_sh)

# Split main_title by a colon and assign it to two columns named title and subtitle 
books_sh[['title', 'subtitle']] = books_sh['main_title'].str.split(':', expand=True)

# Split version by a space and assign the second element to the column named volume 
books_sh['volume'] = books_sh['version'].str.split(' ').str.get(1)

# Drop the main_title and version columns modifying books_sh
books_sh.drop(['main_title', 'version'], axis=1, inplace=True)

# Reshape using title, subtitle and volume as index, name feature the new variable from columns starting with number, separated by undescore and ending in words
sh_long = pd.wide_to_long(books_sh, stubnames='number', i=['title', 'subtitle', 'volume'], 
                          j='feature', sep='_', suffix='\w+')

# Print sh_long
print(sh_long)

## Stack the calls!

# Predefined list to use as index
new_index = [['California', 'California', 'New York', 'Ohio'], 
             ['Los Angeles', 'San Francisco', 'New York', 'Cleveland']]

# Create a multi-level index using predefined new_index
churn_new = pd.MultiIndex.from_arrays(new_index, names=['state', 'city'])

# Print churn_new
print(churn_new)

# Predefined list to use as index
new_index = [['California', 'California', 'New York', 'Ohio'], 
             ['Los Angeles', 'San Francisco', 'New York', 'Cleveland']]

# Create a multi-level index using predefined new_index
churn_new = pd.MultiIndex.from_arrays(new_index, names=['state', 'city'])

# Assign the new index to the churn index
churn.index = churn_new

# Print churn
print(churn)

# Predefined list to use as index
new_index = [['California', 'California', 'New York', 'Ohio'], 
             ['Los Angeles', 'San Francisco', 'New York', 'Cleveland']]

# Create a multi-level index using predefined new_index
churn_new = pd.MultiIndex.from_arrays(new_index, names=['state', 'city'])

# Assign the new index to the churn index
churn.index = churn_new

# Reshape by stacking churn DataFrame
churn_stack = churn.stack()

# Print churn_stack
print(churn_stack)

## Phone directory index

# Set state and city as index modifying the DataFrame
churn.set_index(['state', 'city'], inplace=True)

# Print churn
print(churn)

# Reshape by stacking the second level
churn_stack = churn.stack(level=1)

# Print churn_stack
print(churn_stack)

## Text me!

# Stack churn by the time column level
churn_time = churn.stack(level='time')

# Print churn_time
print(churn_time)

# Stack churn by the feature column level
churn_feature = churn.stack(level='feature')

# Print churn_feature
print(churn_feature)

## International caller

# Reshape the churn DataFrame by unstacking
churn_unstack = churn.unstack()

# Print churn_unstack
print(churn_unstack)

# Reshape churn by unstacking the first row level
churn_first = churn.unstack(level=0)

# Print churn_zero
print(churn_first)

# Reshape churn by unstacking the second row level
churn_second = churn.unstack(level=1)

# Print churn_second
print(churn_second)

## Call another time

# Unstack the time level from churn
churn_time = churn.unstack(level='time')

# Print churn_time
print(churn_time)

# Sort the index in descending order
churn_time = churn.unstack(level='time').sort_index(ascending=False)

# Print churn_time
print(churn_time)

## Organizing your voicemail

# Unstack churn by type level
churn_type = churn.unstack(level='type')

# Stack churn_final using the first column level
churn_final = churn_type.stack(level=0)

# Print churn_final
print(churn_final)

## Swap your SIM card

# Switch the first and third row index levels in churn
churn_swap = churn.swaplevel(0, 2)

# Print churn_swap
print(churn_swap)

# Reshape by unstacking the last row level 
churn_unstack = churn_swap.unstack()

# Print churn_unstack
print(churn_unstack)

## Two many calls

# Unstack the first and second row level of churn
churn_unstack = churn.unstack(level=[0, 1])

# Print churn_unstack
print(churn_unstack)

# Stack the resulting DataFrame using plan and year
churn_py = churn_unstack.stack(['plan', 'year'])

# Print churn_py
print(churn_py)

# Switch the first and second column levels
churn_switch = churn_py.swaplevel(0, 1, axis=1)

# Print churn_switch
print(churn_switch)

## A missed phone call

# Unstack churn level and fill missing values with zero
churn = churn.unstack(level='churn', fill_value=0)

# Sort by descending voice mail plan and ascending international plan
churn_sorted = churn.sort_index(level=["voice_mail_plan", "international_plan"], 
                                ascending=[False, True])

# Print final DataFrame and observe pattern
print(churn_sorted)

## Don't drop the stack

# Stack the level type from churn
churn_stack = churn.stack(level='type')

# Fill the resulting missing values with zero
churn_fill = churn_stack.fillna(0)

# Print churn_fill 
print(churn_fill)

# Stack the level scope without dropping rows with missing values
churn_stack = churn.stack(level='scope', dropna=False)

# Fill the resulting missing values with zero 
churn_fill = churn_stack.fillna(0)

# Print churn_fill
print(churn_fill)

## Less fast food, please!

# Unstack the first level and calculate the mean of the columns
obesity_general = obesity.unstack(level=0).mean(axis=1)

# Print obesity_general
print(obesity_general)

# Unstack the second level and calculate the mean of the columns
obesity_mean = obesity.unstack(level=1).mean(axis=1)

# Print obesity_mean
print(obesity_mean)

# Unstack the third level and calculate the difference between columns
obesity_variation = obesity.unstack(level=2).diff(axis=1)

# Print obesity_variation
print(obesity_variation)

## Only going up

# Stack obesity, get median of columns and unstack again
median_obesity = obesity.stack().median(axis=1).unstack()

# Print median_obesity
print(median_obesity)

# Stack the first level, get sum, and unstack the second level
obesity_sum = obesity.stack(level=0).sum(axis=1).unstack(level=1)

# Print obesity_max
print(obesity_sum)

## A group analysis

# Stack country level, group by country and get the mean 
obesity_mean = obesity.stack(level='country').groupby('country').mean()

# Print obesity_mean
print(obesity_mean)

# Stack country level, group by country and get the median 
obesity_median = obesity.stack(level='country').groupby('country').median()

# Print obesity_mean
print(obesity_median)

## Merge it all

# Explode the values of bounds to a separate row
obesity_bounds = obesity['bounds'].explode()

# Print obesity_bounds
print(obesity_bounds)

# Merge obesity_bounds with country and perc_obesity columns of obesity using the indexes
obesity_final = obesity[['country', 'perc_obesity']].merge(obesity_bounds, 
                                                           right_index=True, 
                                                           left_index=True)

# Print obesity_final
print(obesity_final)

## Explode the bounds

# Transform the list-like column named bounds 
obesity_explode = obesity.explode('bounds')

# Modify obesity_explode by resetting the index
obesity_explode.reset_index(drop=True, inplace=True)

# Print obesity_explode
print(obesity_explode)

## The good old split

# Split the columns bounds using a hyphen as delimiter
obesity_split = obesity['bounds'].str.split('-')

# Print obesity_split
print(obesity_split)

# Assign the result of the split to the bounds column
obesity_split = obesity.assign(bounds=obesity['bounds'].str.split('-'))

# Print obesity_split
print(obesity_split)

# Transform the column bounds in the obesity DataFrame
obesity_split = obesity.assign(bounds=obesity['bounds'].str.split('-')).explode('bounds')

# Print obesity_split
print(obesity_split)

## Nested movies

# Import the json_normalize function
from pandas import json_normalize

# Normalize movies and separate the new columns with an underscore
movies_norm = json_normalize(movies, sep='_')

# Reshape using director and producer as index, create movies from column starting from features
movies_long = pd.wide_to_long(movies_norm, stubnames='features', 
                              i=['director', 'producer'], j='movies', 
                              sep='_', suffix='\w+')

# Print movies_long
print(movies_long)

## A complex film

# Normalize the JSON contained in movies
normalize_movies = json_normalize(movies)

# Print normalize_movies
print(normalize_movies)

# Specify the features column as the list of records 
normalize_movies = json_normalize(movies, 
                                  record_path='features')

# Print normalize_movies
print(normalize_movies)

# Specify director and producer to use as metadata for each record 
normalize_movies = json_normalize(movies, 
                                  record_path='features', 
                                  meta=['director', 'producer'])

# Print normalize_movies
print(normalize_movies)

## Un-nesting birds

# Define birds reading names and bird_facts lists into names and bird_facts columns 
birds = pd.DataFrame({'names': names, 'bird_facts': bird_facts})

# Print birds
print(birds)

# Define birds reading names and bird_facts lists into names and bird_facts columns
birds = pd.DataFrame(dict(names=names, bird_facts=bird_facts))

# Apply the function json.loads function to the bird_facts column
data_split = birds['bird_facts'].apply(json.loads).apply(pd.Series)

# Print birds
print(data_split)

# Define birds reading names and bird_facts lists into names and bird_facts columns 
birds = pd.DataFrame(dict(names=names, bird_facts=bird_facts))

# Apply to bird_facts column the function loads from json module
data_split = birds['bird_facts'].apply(json.loads).apply(pd.Series)

# Remove the bird_facts column from birds
birds = birds.drop(columns=['bird_facts'])

# Print birds
print(birds)

# Define birds reading names and bird_facts lists into names and bird_facts columns
birds = pd.DataFrame(dict(names=names, bird_facts=bird_facts))

# Apply to bird_facts column the function loads from json module
data_split = birds['bird_facts'].apply(json.loads).apply(pd.Series)

# Remove the bird_facts column from birds
birds = birds.drop(columns='bird_facts')

# Concatenate the columns of birds and data_split
birds = pd.concat([birds, data_split], axis=1)

# Print birds
print(birds)

## Don't dump the bird

# Apply json.loads to the bird_facts column and transform it to a list
birds_facts = birds['bird_facts'].apply(json.loads).tolist()

# Print birds_facts
print(birds_facts)

# Apply json.loads to the bird_facts column and transform it to a list
birds_facts = birds['bird_facts'].apply(json.loads).to_list()

# Convert birds_facts into a JSON 
birds_dump = json.dumps(birds_facts)

# Print birds_dump
print(birds_dump)

# Apply json.loads to the bird_facts column and transform it to a list 
birds_facts = birds['bird_facts'].apply(json.loads).to_list()

# Convert birds_facts into a JSON 
birds_dump = json.dumps(birds_facts)

# Read the JSON birds_dump into a DataFrame 
birds_df = pd.read_json(birds_dump)

# Print birds_df
print(birds_df)

# Apply json.loads to the bird_facts column and transform it to a list 
birds_facts = birds['bird_facts'].apply(json.loads).to_list()

# Convert birds_fact into a JSON 
birds_dump = json.dumps(birds_facts)

# Read the JSON birds_dump into a DataFrame
birds_df = pd.read_json(birds_dump)

# Concatenate the 'names' column of birds with birds_df 
birds_final = pd.concat([birds['names'], birds_df], axis=1)

# Print birds_final
print(birds_final)

## The final reshape