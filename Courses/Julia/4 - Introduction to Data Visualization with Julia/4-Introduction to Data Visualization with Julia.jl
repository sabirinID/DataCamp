# Introduction to Data Visualization with Julia

## Interpreting a plot

press 3 # Correct! The plot shows an apparent increase in volume around March 2020.

## Installing Plots.jl

press 2 # Perfect! You correctly identified how to install Plots.jl.

## Price goes up

# Import Plots.jl
using Plots

# Load dataset
iwf = DataFrame(CSV.File("iwf.csv"))

# Create a line plot
plot(
	# Pass the columns
	iwf.price_date,
	iwf.high
)

## Price changes

# Import Plots.jl
using Plots

# Define price change
iwf[!, "price_change"] = iwf.close - iwf.open
iwm[!, "price_change"] = iwm.close - iwm.open

# Scatter plot of price changes
scatter(
	iwm.price_change,
	iwf.price_change
)

## Trading volume

# Volume line plot
plot(
	iwf.price_date,
    iwf.volume,
    # Add a title
    title="IWF Daily Traded Volume",
    # Include axis labels
    xlabel="Date",
    ylabel="Traded Volume"
)

## Multiple line plots

# Plot the QQQ closing price, give it a title
plot(
	qqq.price_date,
	qqq.close,
    label="QQQ",
    linecolor=:green,
    linewidth=2
)
# Give the plot a title
title!("ETF Funds Close Prices")
xlabel!("Date")
ylabel!("Price (USD)")

# Plot the IWF closing price
plot(
	iwf.price_date,
	iwf.close,
    label="IWF",
    linecolor=:pink1,
    linewidth=2
)
title!("ETF Funds Close Prices")
xlabel!("Date")
ylabel!("Price (USD)")

# Plot both line plots in the same figure
plot(
	qqq.price_date,
	[qqq.close iwf.close],
    label=["QQQ" "IWF"],
    linecolor=[:green :pink1],
    linewidth=2
)
title!("ETF Funds Close Prices")
xlabel!("Date")
ylabel!("Price (USD)")

## Correlated ETF funds

# Make a scatter plot
scatter(
	spy.volume,
    qqq.volume,
    smooth=true,
    linewidth=2.5,
    linecolor=:black,
    label=false,
    markercolor=:pink
)
# Add a title to the figure
title!("Correlation Between SPY & QQQ Traded Volumes")
# Label the x and y axes
xlabel!("SPY Daily Traded Volume")
ylabel!("QQQ Daily Traded Volume")

## Multiple line plots!

# Plot the closing price
scatter(spy.price_date,
        spy.close,
        markercolor=:yellow,
        label="Closing Price")
title!("Closing Price of SPY")
xlabel!("Date")
ylabel!("Price (USD)")
# Add the moving average plot
plot!(spy.price_date,
      spy.close_ma,
      linewidth=3,
      linecolor=:purple1,
      label="7-day Average")

## Potato prices

# Plot a histogram
histogram(
	potato[:, "Retail Price"],
    # Hide the label 
    label=false,
    # Choose color
    color=:orange3
)
title!("Potato Prices in India")
xlabel!("Price (Rupees)")
ylabel!("Frequency")

# Plot a histogram
histogram(
	potato[:, "Retail Price"],
    label=false,
    color=:orange3,
    # Number of bins
    bins=10
)
title!("Potato Prices in India")
xlabel!("Price (Rupees)")
ylabel!("Frequency")

# Plot a histogram
histogram(
	potato[:, "Retail Price"],
    # Hide the label 
    label=false,
    # Choose color
    color=:orange3,
    # Number of bins
    bins=range(0, 75, 25)
)
title!("Potato Prices in India")
xlabel!("Price (Rupees)")
ylabel!("Frequency")

## Rajasthan normalized

# Plot histogram
histogram(
	rajasthan."Retail Price",
    label="Rajasthan",
)

# Plot histogram
histogram(
	rajasthan."Retail Price",
    label="Rajasthan",
    # Normalize data
    normalize=true,
)

histogram(
	rajasthan."Retail Price",
    label="Rajasthan",
    normalize=true,
)
# Add density plot
density!(
	rajasthan."Retail Price",
    color=:black,
    # Set the line width
    linewidth=2,
    label=false
)
xlabel!("Price (Rupees)")
ylabel!("Probability")

## Instant and powdered coffee

# import StatsPlots
using StatsPlots

# Create grouped histogram
groupedhist(
	coffee."Retail Price",
    # Define the group
    group=coffee.Variety,
    # Choose colors
    color=[:sandybrown :brown]
)
title!("Cofee Prices in India")
xlabel!("Price (Rupees)")
ylabel!("Frequency")

## Stacking states

# Create grouped histogram
groupedhist(
	commodities."Retail Price",
    # Define groups and colors
	group=commodities.State,
    color=[:limegreen :fuchsia :orange1],
    # Stack the bars
    bar_position=:stack
)
title!("Commodity Prices in India")
xlabel!("Price (Rupees)")
ylabel!("Frequency")

## Price deviations

# Group by commodities
grouped = groupby(product, :Commodity)
# Calculate standard deviation
deviations = combine(grouped, :"Retail Price" => std)

# Create bar chart
bar(
    deviations.Commodity,
    deviations."Retail Price_std",
    label=false,
    # Permute the axes
    permute=(:x, :y),
)
title!("Deviation of Prices")
xlabel!("Standard Deviation (Rupees)")

## Saree prices

# Create grouped bar chart
groupedbar(
	saree_mean_prices.State,
    saree_mean_prices."Retail Price",
    # Group by Variety
    group=saree_mean_prices.Variety,
    color=[:lightpink1 :purple3]
)
title!("Saree Prices per State")
ylabel!("Average Unit Price (Rupees)")

# Create grouped bar chart
groupedbar(
	saree_mean_prices.State,
    saree_mean_prices."Retail Price",
    # Group by variety
    group=saree_mean_prices.Variety,
    color=[:lightpink1 :purple3],
    # Stack those bars
    bar_position=:stack,
)
title!("Saree Prices per State")
ylabel!("Average Unit Price (Rupees)")

## Them apples

# Plot time series
plot(
	apple.Date,
    apple."Retail Price",
    color=:red2,
    linewidth=2.5,
    label="Apple"
)
ylabel!("Price (Rupees)")

# Date column to Date type
apple.Date = Date.(apple.Date, dateformat"y-U")

# Plot time series
plot(
	# Price versus date
	apple.Date,
    apple."Retail Price",
    color=:red2,
    linewidth=2.5,
    label="Apple"
)
ylabel!("Price (Rupees)")

# Date column to Date type
apple.Date = Date.(apple.Date, dateformat"y-U")
# Sort by date
apple = sort(apple, :Date)

# Plot time series
plot(
	# Price versus date
	apple.Date,
    apple."Retail Price",
    color=:red2,
    linewidth=2.5,
    label="Apple"
)
ylabel!("Price (Rupees)")

## (Super)fine rice

# Plot both time series
plot(fine.Date,
    [fine."Retail Price" superfine."Retail Price"],
    label=["Fine Rice" "Superfine Rice"],
    linewidth=2)
ylabel!("Price (Rupees)")

# Plot both time series
plot(fine.Date,
    [fine."Retail Price" superfine."Retail Price"],
    label=["Fine Rice" "Superfine Rice"],
    linewidth=2)
ylabel!("Price (Rupees)")
# Find when prices match
condition = fine."Retail Price" .== superfine."Retail Price"
common_price = fine[condition, :]

# Plot both time series
plot(fine.Date,
    [fine."Retail Price" superfine."Retail Price"],
    label=["Fine Rice" "Superfine Rice"],
    linewidth=2)
ylabel!("Price (Rupees)")
# Find when prices match
condition = fine."Retail Price" .== superfine."Retail Price"
common_price = fine[condition, :]
# Add annotation
annotate!(
	# Coordinates
	common_price.Date, common_price."Retail Price" .+ 7.5,
    # Text
    "Prices\nmatch", annotationfontsize=10)

## Boxing states

# Create box plot
boxplot(
	commodities.State,
    commodities."Retail Price",
    # Customize it
    color=:mediumseagreen,
    label=false,
)
ylabel!("Price (Rupees)")

# Create box plot
boxplot(
	commodities.State,
    commodities."Retail Price",
    # Customize it
    color=:mediumseagreen,
    label=false,
    # Hide outliers
    outliers=false,
)
ylabel!("Price (Rupees)")

## Price distributions

# Create violin plot
violin(
	product.Commodity,
    product."Retail Price",
    # Hide line
    linewidth=0,
    # Set the color
    color=:royalblue1,
    label=false,
)
title!("Product Prices in India")
ylabel!("Price (Rupees)")

## Tomato seasons

# Create a violin plot
violin(tomatoes.Month,
	tomatoes[:, "Retail Price"],
    label=false,
    xticks=(1:12, month_labels),
   	color=:crimson)
# Add a box plot to the figure
boxplot!(tomatoes.Month,
	tomatoes[:, "Retail Price"],
	label=false,
    outliers=false,
    color=:turquoise3)
ylabel!("Price (Rupees)")

## Saving to file

press 4 # Perfect! You are ready to save your beautiful creations!

## Theme showdown

# Plot a histogram
histogram(
	streaming.Age,
    label=false
)
title!("Age of Survey Respondents")
xlabel!("Age")
ylabel!("Frequency")

# Set the theme
theme(:ggplot2, color=:seagreen)

# Plot a histogram
histogram(
	streaming.Age,
    label=false
)
title!("Age of Survey Respondents")
xlabel!("Age")
ylabel!("Frequency")

# Set the theme
theme(
	:dracula,
    linewidth=0,
    # Remove the axis lines
    framestyle=:grid
)

histogram(
	streaming.Age,
    label=false
)
title!("Age of Survey Respondents")
xlabel!("Age")
ylabel!("Frequency")

## Pump it up

# Choose theme
theme(:vibrant)

# Create scatter plot
scatter(
	streaming.Age, streaming.BPM,
    label=false,
    # Marker attributes
    markersize=4, markercolor=:purple,
    markershape=:pentagon,
    # Opacity
    alpha=0.5,
)
xlabel!("Age")
ylabel!("BPM")

## Music effects on depression

theme(:ggplot2)

density(streaming.Depression,
    group=streaming.Music_effects,
    # Set line attributes
    linewidth=4,
    linestyle=[:solid :dash :dot],
    # Customize the legend
    legend_title="Music effect",
    legend_position=:bottom)
title!("Effect of Music on Depression")
xlabel!("Depression Level")
ylabel!("Probability")
# x-axis bounds
xlims!(0,10)

## Self-reported conditions

# Set theme
theme(
	:wong, framestyle=:grid, label=false, alpha=0.75
)

# Create violin plot
violin(conditions.Condition,
    conditions.Value,
    linewidth=0)
# Add box plot
boxplot!(conditions.Condition,
    conditions.Value,
    linewidth=2, 
    linecolor=:midnightblue)
ylabel!("Self-reported Level")

## Composers mental health

# Number of grouped rows 
grouped = groupby(streaming,
    ["Music_effects", "Composer"])
counts = combine(grouped, nrow => :Count)

# Number of grouped rows 
grouped = groupby(streaming,
    ["Music_effects", "Composer"])
counts = combine(grouped, nrow => :Count)

theme(:bright)

bar(counts.Music_effects, counts.Count,
	group=counts.Composer, color=[:seagreen3 :purple],
    linewidth=0, legend_title="Composer",
    # Set layout
    layout=2,
    # Y-axis labels
    ylabel=["Frequency" ""])
xlabel!("Music Effects")

## Streaming while working

theme(:vibrant)
# Box plots
boxplot(
	streaming.While_working, streaming.Hours_per_day,
    group=streaming.Music_effects,
    outliers=false, linewidth=1,
    color=[:darkorange :springgreen2 :slateblue3],
    # Grid layout
    layout=(1, 3),
    # Axis labels
    xlabel=["" "Music While Working" ""],
    ylabel=["Hours Per Day" "" ""]
)
# Set y-axis limits
ylims!(0, 10)

## Favorite genres

theme(:mute, markeralpha=0.25)
# Bar plots
p1 = bar(counts.Fav_genre, counts.Count,
	group=counts.Foreign_languages, linewidth=0,
    layout=2, alpha=0.75, ylabel="Frequency",
    color=[:dodgerblue3 :brown1],
    legend_title="Foreign Languages")

theme(:mute, markeralpha=0.25)
# Bar plots
p1 = bar(counts.Fav_genre, counts.Count,
	group=counts.Foreign_languages, linewidth=0,
    layout=2, ylabel="Frequency",
    color=[:dodgerblue3 :brown1],
    legend_title="Foreign Languages")
# Scatter plots
p2 = scatter(streaming.Fav_genre, streaming.Age,
    group=streaming.Fav_genre, label=false, ylabel="Age")
p3 = scatter(streaming.Fav_genre, streaming.BPM,
    group=streaming.Fav_genre, label=false, ylabel="BPM")

theme(:mute, markeralpha=0.25)
# Bar plots
p1 = bar(counts.Fav_genre, counts.Count,
	group=counts.Foreign_languages, linewidth=0,
    layout=2, ylabel="Frequency",
    color=[:dodgerblue3 :brown1],
    legend_title="Foreign Languages")
# Scatter plots
p2 = scatter(streaming.Fav_genre, streaming.Age,
    group=streaming.Fav_genre, label=false, ylabel="Age")
p3 = scatter(streaming.Fav_genre, streaming.BPM,
    group=streaming.Fav_genre, label=false, ylabel="BPM")
# Join the plots
layout = @layout [a; b c]
plot(p1, p2, p3, layout=layout)

## Spotify as therapy

# Define series recipe
@recipe function f(::Type{Val{:my_box}}, x, y, z)
    seriestype := :box
    framestyle := :box
    grid := :off
    label := false
    outliers := false
    fillcolor := :plum
    linecolor := :darkorchid
end

@recipe function f(::Type{Val{:my_box}}, x, y, z)
    seriestype := :box
    framestyle := :box
    grid := :off
    label := false
    outliers := false
    fillcolor := :plum
    linecolor := :darkorchid
end
# Use series recipe
plot(spotify.Music_effects, spotify.Age, 
	seriestype=:my_box)
xlabel!("Music Effect")
ylabel!("Age")

@recipe function f(::Type{Val{:my_box}}, x, y, z)
    seriestype := :box
    framestyle := :box
    grid := :off
    label := false
    outliers := false
    fillcolor := :plum
    linecolor := :darkorchid
end
# Define plotting function
@shorthands my_box
# Use series plotting function
my_box(spotify.Music_effects, spotify.Depression)
xlabel!("Music Effect")
ylabel!("Self-reported Depression")

## Log scatter recipe

# Series recipe
@recipe function f(::Type{Val{:logscatter}}, x, y, z)
    seriestype := :scatter
    framestyle := :box
    label := false
    yscale := :log10
    markershape := :star4
    markercolor := :purple3
    markersize := 5
    markeralpha := 0.5
end

# Define plotting function
@shorthands logscatter
logscatter(streaming.Age, streaming.Hours_per_day)

## Upcharging older smokers

# Use recipe
@df insurance scatter(
	# Pass columns
	:Age,
    :Charges,
    group=:Smoker,
    legend_title="Smoker",
	# Customize markers
    markershape=:rect,
    markersize=5,
    markercolor=[:lightseagreen :crimson]
)
xlabel!("Age")
ylabel!("Insurance Premium (USD)")

## Upcharging heavier smokers

# Import the necessary package
using StatsPlots

# Use recipe
@df insurance scatter(
	# Pass columns
	:BMI,
    :Charges,
    group=:Smoker,
    legendtitle="Smoker")
xlabel!("Body Mass Index (BMI)")
ylabel!("Insurance Premium (USD)")

## BMI versus age

grouped = groupby(insurance, :Age)
mean_bmis = combine(grouped, :BMI => mean)

# Scatter plot
scatter(mean_bmis.Age, mean_bmis.BMI_mean,
	label=false, smooth=true,
    linewidth=3, linecolor=:maroon1)
xlabel!("Age")
ylabel!("Body Mass Index (BMI)")

grouped = groupby(insurance, :Age)
mean_bmis = combine(grouped, :BMI => mean)

# DataFrame recipe
@df mean_bmis scatter(:Age, :BMI_mean,
	label=false, smooth=true,
    linewidth=3, linecolor=:maroon1)
xlabel!("Age")
ylabel!("Body Mass Index (BMI)")

# Define chain
@chain insurance begin
	groupby(:Age)
    combine(:BMI => mean)
    @df scatter(:Age, :BMI_mean,
		label=false, smooth=true,
    	linewidth=3, linecolor=:maroon1)
end
xlabel!("Age")
ylabel!("Body Mass Index (BMI)")

## Slicing by region

# Grid of scatter plots
@df insurance scatter(
	:BMI,
    :Charges,
    group=:Region,
    markersize=2,
    color=[:darkorange :dodgerblue4 :deepskyblue :deeppink],
    legend_position=:topleft,
    layout=(2, 2),
)
xlabel!("BMI")
ylabel!("Premium (USD)")

# Three-dimensional scatter plot
@df insurance scatter(
	:Region,
    :BMI,
    :Charges,
    # Group by region
    group=:Region,
    markersize=2,
    legend_position=:topright
)
ylabel!("BMI")
zlabel!("Insurance Premium (USD)")

## Aging and having kids

# Create 2d histogram
@df insurance histogram2d(
	# Pass column names
	:Age,
    :Children,
    fillcolor=:haline,
    # Fill empty bins
    show_empty_bins=true
)
xlabel!("Age")
ylabel!("Number of Children")

## BMI per region

# Violin plot
@df insurance violin(
	:Region, :BMI,
    linewidth=0, fillcolor=:magenta4,
    label="Distribution"
)

# Violin plot
@df insurance violin(
	:Region, :BMI,
    linewidth=0, fillcolor=:magenta4,
    label="Distribution"
)
# Add scatter plot
@df insurance scatter!(
	:Region, :BMI,
    markersize=4, markercolor=:lightsalmon,
    markershape=:hexagon, alpha=0.25,
    label="Data Point"
)
xlabel!("Region")
ylabel!("Body Mass Index (BMI)")

## Smokers by age

# Create histograms
@df insurance histogram(
	:Age,
    group=:Smoker,
    color=[:mediumorchid :seagreen],
    legend_title="Smoker",
    legend_position=:outertopright,
    # Set layout
    layout=(2,1),
    # Label x-axes
    xlabel=["" "Age"]
)
ylims!(0, 120)
ylabel!("Frequency")

## Region premiums

colors = [:slategray1 :springgreen4 :deeppink :darkviolet]

@chain insurance begin
	# Change charge units
    transform(:Charges
    	=> ByRow(x -> x/1000) => :Charges)
	# Create histograms
    @df histogram(:Charges, group=:Region,
    	layout=(2,2),
        color=colors,
        # Set axis labels
        xlabel=["" "" "Premium (kUSD)" "Premium (kUSD)"],
        ylabel=["Frequency" "" "Frequency" ""])
end
ylims!(0, 120)

## Wrap-up