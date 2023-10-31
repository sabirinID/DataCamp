# Data Manipulation in Julia

## Symbols vs. Strings

# Select the species column using symbol
penguins[:, :species]

# Select body mass g column using string
penguins[:, "body mass g"]

# Select body mass g column using symbol
penguins[:, Symbol("body mass g")]

## Describe it to me

# Describe chocolates using only the min, max, and number of missing values
describe(chocolates, :min, :max, :nmissing)

# Describe chocolates using the data types and the sum of its values
describe(chocolates, :eltype ,sum => :total)

## Column selection

# Select species, island, and sex columns
select(penguins, :species, :island, :sex)

# Select species, culmen_length_mm, and body_mass_g using position
select(penguins, 1, 3, 6)

# Select columns
select(penguins, "species", :sex, 2)

## Selecting patterns

# Select columns starting with the letter r and columns ending with location
reviews_locations = select(chocolates, Cols(startswith("r")), Cols(endswith("location")))

# Sort reviews_locations using the rating column
sort!(reviews_locations, :rating, rev = true)

println(first(reviews_locations, 5))

## Regular penguins

# Select all columns containing measurements in millimeters
select(penguins, r"mm")

## Flipper distribution

# Make a histogram of flipper lengths with 5 bins
histogram(penguins.flipper_length_mm, bins = 5)

# Make a histogram of flipper lengths with 15 bins
histogram(penguins.flipper_length_mm, bins = 15)

## Rating vs. cocoa percentages

# Make a scatter plot of cocoa vs. ratings
scatter(chocolates.rating, chocolates.cocoa)

# Add title
title!("Cocoa percentage vs. rating")

## Plotting minimum wages over time

# Plot the median wage and label it 
plot(wages_stats.year, wages_stats.median, label="median")

# Plot the maximum and label it 
plot!(wages_stats.year, wages_stats.max, label="max")

# Plot the minimum and label it
plot!(wages_stats.year, wages_stats.min, label="min")

# Add a title
title!("Trends in minimum wages in the US")

# Add x and y labels
xlabel!("Year")
ylabel!("Inflation-adjusted Minimum Wage (USD)")

## Penguin drop

# Drop the column sex
select!(penguins, Not(:sex))

# Safely drop the sex column
select(penguins, Not(Cols(==("sex"))))

## Reordering of wages

# Print the first line
println(first(wages))

# Move the effective_min_wage_2020_dollars column after the region column
select!(wages, :year, :state, :region, :effective_min_wage_2020_dollars, :)

# Move the CPI_average column to the end
select!(wages, :year, :state, :region, :effective_min_wage_2020_dollars, Not(:CPI_average), :CPI_average)

# Print the first line
println(first(wages))

## Using select()

# Select columns and rename the region column
select(wages, :year, :region => :us_regions, r"2020")

# Calculate mean, minimum, and maximum of state_min_wage
select(wages, :state_min_wage .=> [mean, minimum, maximum])

## Penguin transformations

# Add new columns containing the median values for flipper_length_mm and body_mass_g
transform(penguins, [:flipper_length_mm, :body_mass_g] .=> median)

## Combining chocolates

# Compute mean of rating, median of cocoa, and std of REF
combine(chocolates, :rating => mean, :cocoa => median, :REF => std)

## What shall you use?

# Add column containing the median of CPI_average
transform!(wages, :CPI_average => median)

# Select CPI_average_median as med_CPI, year, federal_min_wage column
select!(wages, :CPI_average_median => :med_CPI, :year, :federal_min_wage => sum)

# Create a 1x1 DataFrame containing the maximum of the year column
combine(wages, :year => maximum)

## Copy the capitals

# Create new column by referencing
wages.capital = capitals

capitals[9] = "N/A"

println(wages[9,:])

# Create new column by copying
wages[:, :capital] = capitals

capitals[9] = "N/A"

println(wages[9,:])

## Chocolate percentages

# Create a new column for cocoa percentage 
transform!(chocolates, "cocoa_percentage" => ByRow(x -> parse(Float64, x[1:end-1])) => :cocoa)

# Drop the cocoa_percentage column
select!(chocolates, Not(:cocoa_percentage))

# Reorder columns
select!(chocolates, :cocoa, :rating, :location, :bean_location)

println(first(chocolates, 5))

## Wages multiple ways

# Group wages by region
groupby(wages, :region)

# Group wages by region and year
groupby(wages, [:region, :year])

# Group wages by year and then region
groupby(wages, [:year, :region])

## Penguin group counts

# Create penguin_species
penguin_species = groupby(penguins, :species)

# Count the observations per group and rename the column, then sort
sort(combine(penguin_species, nrow => :penguins_observed), :penguins_observed, rev = true)

## Unique chocolate beans

# Find unique values of bean_type
unique(chocolates.bean_type)

## Duplicate rows or not?

10 # Correct! There are 10 duplicate rows in the corrupted penguins dataset. Good job on pointing it out!

## Penguin characteristics

# Create penguin_species
penguin_species = groupby(penguins, :species)

# Calculate the median of columns
combine(penguin_species, [:flipper_length_mm, :culmen_length_mm, :body_mass_g] .=> median)

## Chocolate location vs. rating

# Create choc_groups
choc_groups = groupby(chocolates, [:company_location, :company])

# Calculate minimum, median, maximum of rating, rename the new columns and save the result
choc_groups_stats = combine(choc_groups, :rating .=> [minimum, median, maximum] .=> [:min_rating, :med_rating, :max_rating])

# Sort choc_groups_stats by med_rating in reverse
sort(choc_groups_stats, :med_rating, rev = true)

## Reshaping wages

# Make a pivot table 
unstack(wages, [:state, :region], :year, :effective_min_wage_2020_dollars)

## Chocolate location pivot

# Sort chocolates by cocoa column
sort!(chocolates, :cocoa)

# Create a pivot table and save it
chocolates_pivot = unstack(chocolates, :company_location, :cocoa_rounded, :rating, combine = median)

# Replace missing values with empty string
chocolates_pivot = unstack(chocolates, :company_location, :cocoa_rounded, :rating, combine=median, fill = "")

# Sort chocolates_pivot by company_location
sort(chocolates_pivot, :company_location)

## Chaining chocolates

select(chocolates, :company, :cocoa_percentage, :rating)

# Rewrite the previous code as a macro
@chain chocolates begin
	select(:company, :cocoa_percentage, :rating)
end

transform(select(chocolates, :company, :cocoa_percentage, :rating),
	:cocoa_percentage => ByRow(x -> parse(Float64, x[1:end-1])) => :cocoa)

# Rewrite the previous code as a macro
@chain chocolates begin
	select(:company, :cocoa_percentage, :rating)
    transform(:cocoa_percentage => ByRow(x -> parse(Float64, x[1:end-1])) => :cocoa)
end

grouped_choc = groupby(chocolates, :company)
combine(grouped_choc, :rating => median)

# Rewrite as macro
@chain chocolates begin
	groupby(:company)
    combine(:rating => median)
end

grouped_choc = groupby(chocolates, :company)
combine(grouped_choc, :rating => median)

# Rewrite as macro
company_ratings = @chain chocolates begin
	groupby(:company)
    combine(:rating => median)
end

## Penguin plotting in chain

@chain penguins begin
	
    # Group by species
    groupby(:species)
    
    # Create a scatter plot for each species group
    @aside scatter(_[1].flipper_length_mm, _[1].body_mass_g, label="Adelie")
    @aside scatter!(_[2].flipper_length_mm, _[2].body_mass_g, label="Chinstrip")
    scatter!(_[3].flipper_length_mm, _[3].body_mass_g, label="Gentoo", legend=:topleft)
    
    xlabel!("Flipper length in mm")
    ylabel!("Body mass in g")
end

## Minimum wage by region

# Create the chain macro
regions_wage = @chain wages begin
	
    # Group by region and year
    groupby([:region, :year])

	# Calculate the median wage per region per year
    combine("effective_min_wage_2020_dollars" => median => :median_effective_wage_2020)

	# Reshape the result
    unstack(:year, :region, :median_effective_wage_2020)
end

make_plot(regions_wage)

## Decimals and delimiters

# Load the file
choc = DataFrame(CSV.File("choc_dashed.csv", delim = '-'))

# Print the describe function
println(describe(choc))

# Load the file
choc = DataFrame(CSV.File("choc_dataset.csv"))

# Print the description of choc
println(describe(choc))

# Load the file with decimal=
choc = DataFrame(CSV.File("choc_dataset.csv", decimal = ','))

# Print the describe function
println(describe(choc))

## Loading the 80s

# Load wages from the 80s
wages_80s = DataFrame(CSV.File("wages.csv",skipto=614,limit=510))

# Print first and last lines to check
println(first(wages_80s))
println(last(wages_80s))

## Write it down

# Save df
CSV.write("/home/repl/df.csv", df, delim = " ")

## State joins capitals

# Join wages and state_info on state
wages = leftjoin(wages, state_info, on = :state)

# Join wages and state_info_original on state
wages = leftjoin(wages, state_info_original, on = :state => :State)

## Penguin joins

# Use leftjoin on penguins and penguin_data
penguins_ids = leftjoin(penguins, penguin_data, on=[:species => :penguin, :island => :location])

## Dropping missing values

println(size(wages))

# Drop all missing values
dropmissing!(wages)

# Print describe and size functions
println(describe(wages))
println(size(wages))

println(size(wages))

# Drop missing values from effective_min_wage_2020_dollars column
dropmissing!(wages, :effective_min_wage_2020_dollars)

# Print describe and size functions
println(describe(wages))
println(size(wages))

## Replacing rating with median

# Replace missing values in rating column by median
replace!(chocolates.rating, missing => median(skipmissing(chocolates.rating)))

println(describe(chocolates, :nmissing))

## Replacing rating with group median

# Group by company and iterate
for group in groupby(chocolates, :company)

	# Subset each group using ismissing() and the rating column, assign a new value 
	group[ismissing.(group.rating),:rating] .= replace_missing(group, :rating)
end

println(describe(chocolates, :nmissing))

## First steps with flights

# Load datasets
airports = DataFrame(CSV.File("airports.csv"))
airlines = DataFrame(CSV.File("airlines.csv"))
flights = DataFrame(CSV.File("flights.csv"))

# Join flights and airports
flights = leftjoin(flights, airports, on = :origin_airport => :IATA_code)

# Join flights and airlines
flights = leftjoin(flights, airlines, on = :airline => :IATA_code)

# Describe flights and print the result
println(describe(flights))

## Missing delays?

# Drop missing values from departure_delay
dropmissing!(flights, :departure_delay)

# Print describe
println(describe(flights))

## Delays on US airports

# Start a chain macro
@chain flights begin
	# Group by origin airport
    groupby(:airport)

	# Calculate the minimum, median, and maximum departure delay
    combine(:departure_delay .=> [minimum, median, maximum])

	# Sort by median departure delay in descending order
    sort(:departure_delay_median, rev=true)
    
    println(first(_,5))
end

## Wrap-up