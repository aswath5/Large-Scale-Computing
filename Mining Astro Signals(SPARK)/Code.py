from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, min, max

# creating a spark session
spark = SparkSession.builder.appName("PulsarDataAnalysis").getOrCreate()

# reading the pulsar dataset with space-separated values, no header, and inferred schema
df = spark.read.csv('pulsar.dat', sep=" ", header=False, inferSchema=True)

# renaming the columns for better readability
df = df.withColumnRenamed('_c0', 'ascension') \
       .withColumnRenamed('_c1', 'declination') \
       .withColumnRenamed('_c2', 'time') \
       .withColumnRenamed('_c3', 'frequency')

# converting the dataframe into an RDD
rdd = df.rdd

# mapping the RDD to key-value pairs: (rounded frequency, ascension, declination) -> time
rdd = rdd.map(lambda row: ((round(row['frequency'], 0), 
                            round(row['ascension'], 0), 
                            round(row['declination'], 0)), row['time']))

# grouping by the rounded frequency, ascension, and declination
grouped_rdd = rdd.groupByKey()

# counting blips for each group
blip_counts = grouped_rdd.mapValues(len)

# finding the key (frequency, ascension, declination) with the most blips
most_blips = blip_counts.max(key=lambda x: x[1])

# extracting the rounded frequency, ascension, and declination with the most blips
rounded_frequency, rounded_ascension, rounded_declination = most_blips[0]
blip_count = most_blips[1]

# calculating time differences between consecutive blips
time_diff_rdd = grouped_rdd.mapValues(lambda times: list(zip(sorted(times)[:-1], sorted(times)[1:]))) \
                           .flatMapValues(lambda pairs: [t2 - t1 for t1, t2 in pairs])

# getting the time differences for the group with the most blips
time_diffs_for_most_blips = time_diff_rdd.filter(lambda x: x[0] == (rounded_frequency, rounded_ascension, rounded_declination)) \
                                         .map(lambda x: x[1]) \
                                         .collect()

# calculating the average time difference
avg_time_diff = sum(time_diffs_for_most_blips) / len(time_diffs_for_most_blips)

# outputting the results
print(f"Blip Count: {blip_count}")
print(f"Coordinates: ({rounded_ascension}, {rounded_declination})")
print(f"Frequency: {rounded_frequency} MHz")
print(f"Time Differences Between Blips: {time_diffs_for_most_blips}")
print(f"Average Time Difference Between Blips: {avg_time_diff}")

#---------------------------------------------------------------------------------

# now, i want to see which frequencies and coordinates have the most blip counts

# sorting the blip counts in descending order
sorted_blip_counts = blip_counts.sortBy(lambda x: x[1], ascending=False)

# formatting the results
formatted_rdd = sorted_blip_counts.map(lambda x: f"{x[0]}: {x[1]}")

# saving the formatted results to a text file
formatted_rdd.saveAsTextFile('pulsar_blip_counts_output')

print("Results have been saved to the pulsar_blip_counts_output directory.")


#--------------------3-------------------------------------------------------------

# Filtering Data for frequency range: 4447 - 4449, ascension range:  85 - 87, declination range: 67 - 69

rdd = df.rdd

# mapping to key-value pairs: ((rounded frequency, ascension, declination), time) to 5 decimal place
rdd_5dec = rdd.map(lambda row: ((round(row['frequency'], 5), 
                            round(row['ascension'], 5), 
                            round(row['declination'], 5)), row['time']))

# goruping by (rounded frequency, ascension, declination)
grouped_rdd5 = rdd_5dec.groupByKey()

# for each group, lets collect the times and count the occurrences
blip_data = grouped_rdd5.mapValues(lambda times: (list(times), len(list(times))))  

# frequency and coordinate ranges
frequency_range4 = (4447, 4449)
ascension_range4 = (85, 87)
declination_range4 = (67, 69)

# filtering the blip data based on the specified ranges
filtered_blip_data_rdd4 = blip_data.filter(
    lambda x: frequency_range4[0] <= x[0][0] <= frequency_range4[1] and 
               ascension_range4[0] <= x[0][1] <= ascension_range4[1] and 
               declination_range4[0] <= x[0][2] <= declination_range4[1]
)

# flattening the data for sorting
flattened_rdd4 = filtered_blip_data_rdd4.flatMap(
    lambda x: [((x[0], time), 1) for time in x[1][0]]
)

# sorting the flattened data by time
sorted_rdd4 = flattened_rdd4.sortByKey(keyfunc=lambda x: x[1])  

# regrouping to collect times and counts by (frequency, ascension, declination) key
sorted_grouped_rdd4 = sorted_rdd4.map(lambda x: (x[0][0], (x[0][1], 1))) \
                                .groupByKey() \
                                .mapValues(lambda times: (sorted([t for t, _ in times]), len(times)))

# formatting the filtered results to include frequency, ascension, declination, count, and times
formatted_filtered_rdd4 = sorted_grouped_rdd4.map(lambda x: f"{x[0]}: Count = {x[1][1]}, Times = {x[1][0]}")
formatted_filtered_rdd4.saveAsTextFile('filtered_blip_counts_4448')

total_blips4 = filtered_blip_data_rdd4.map(lambda x: x[1][1]).sum()
print(f"Total number of blips: {total_blips4}")


print("Filtered results for 4448 have been saved to the directory.")

#---------------------------------------------------------------------------------

# Filtering Data for frequency range: 3030 - 3032, ascension range:  103 - 105, declination range: 110 - 112

# new frequency and coordinate ranges
frequency_range3 = (3030, 3032)
ascension_range3 = (103, 105)
declination_range3 = (110, 112)

# filtering the blip data based on the specified ranges
filtered_blip_data_rdd3 = blip_data.filter(
    lambda x: frequency_range3[0] <= x[0][0] <= frequency_range3[1] and 
               ascension_range3[0] <= x[0][1] <= ascension_range3[1] and 
               declination_range3[0] <= x[0][2] <= declination_range3[1]
)

# flattening the data for sorting
flattened_rdd3 = filtered_blip_data_rdd3.flatMap(
    lambda x: [((x[0], time), 1) for time in x[1][0]]
)

# sorting the flattened data by time
sorted_rdd3 = flattened_rdd3.sortByKey(keyfunc=lambda x: x[1]) 

# regrouping to collect times and counts by (frequency, ascension, declination) key
sorted_grouped_rdd3 = sorted_rdd3.map(lambda x: (x[0][0], (x[0][1], 1))) \
                                .groupByKey() \
                                .mapValues(lambda times: (sorted([t for t, _ in times]), len(times)))

# formatting the filtered results to include frequency, ascension, declination, count, and times
formatted_filtered_rdd3 = sorted_grouped_rdd3.map(lambda x: f"{x[0]}: Count = {x[1][1]}, Times = {x[1][0]}")
formatted_filtered_rdd3.saveAsTextFile('filtered_blip_counts_3030')

total_blips3 = filtered_blip_data_rdd3.map(lambda x: x[1][1]).sum()
print(f"Total number of blips: {total_blips3}")

print("Filtered results have been saved to the directory.")

#---------------------------------------------------------------------------------

# making this range shorter again for both
# making this now ~ 3 st dev (so 0.6 units range), from the median of these values we got
# because looking at the outputs, I am trying to check if for the frequency around 3030,
# has more blips for the smaller standard deviation

# frequency and coordinate ranges
frequency_range_n4 = (4447.7426, 4448.3426) # from output above: median = 4448.04258
ascension_range_n4 = (86.19724, 86.79724) # median: 86.49724
declination_range_n4 = (67.82535, 68.42535) # median: 68.12535

# filtering the blip data based on the specified ranges
filtered_blip_data_rdd_n4 = blip_data.filter(
    lambda x: frequency_range_n4[0] <= x[0][0] <= frequency_range_n4[1] and 
               ascension_range_n4[0] <= x[0][1] <= ascension_range_n4[1] and 
               declination_range_n4[0] <= x[0][2] <= declination_range_n4[1]
)

# flattening the data for sorting
flattened_rdd_n4 = filtered_blip_data_rdd_n4.flatMap(
    lambda x: [((x[0], time), 1) for time in x[1][0]]
)

# sorting the flattened data by time
sorted_rdd_n4 = flattened_rdd_n4.sortByKey(keyfunc=lambda x: x[1])  

# regrouping to collect times and counts by (frequency, ascension, declination) key
sorted_grouped_rdd_n4 = sorted_rdd_n4.map(lambda x: (x[0][0], (x[0][1], 1))) \
                                .groupByKey() \
                                .mapValues(lambda times: (sorted([t for t, _ in times]), len(times)))

# formatting the filtered results to include frequency, ascension, declination, count, and times
formatted_filtered_rdd_n4 = sorted_grouped_rdd_n4.map(lambda x: f"{x[0]}: Count = {x[1][1]}, Times = {x[1][0]}")

total_blips_n4 = filtered_blip_data_rdd_n4.map(lambda x: x[1][1]).sum()
print(f"Total number of blips with 3 sd 4448: {total_blips_n4}")

# this outputs 35

#--------------------------------------------------------------------------

# similarly for the other

frequency_range_n3 = (3030.38861, 3030.98861) # median: 3030.68861
ascension_range_n3 = (104.20624, 104.80624) # median: 104.50624
declination_range_n3 = (111.12846, 111.72846) # median: 111.42846

# filtering the blip data based on the specified ranges
filtered_blip_data_rdd_n3 = blip_data.filter(
    lambda x: frequency_range_n3[0] <= x[0][0] <= frequency_range_n3[1] and 
               ascension_range_n3[0] <= x[0][1] <= ascension_range_n3[1] and 
               declination_range_n3[0] <= x[0][2] <= declination_range_n3[1]
)

# flattening the data for sorting
flattened_rdd_n3 = filtered_blip_data_rdd_n3.flatMap(
    lambda x: [((x[0], time), 1) for time in x[1][0]]
)

# sorting the flattened data by time
sorted_rdd_n3 = flattened_rdd_n3.sortByKey(keyfunc=lambda x: x[1])  

# regrouping to collect times and counts by (frequency, ascension, declination) key
sorted_grouped_rdd_n3 = sorted_rdd_n3.map(lambda x: (x[0][0], (x[0][1], 1))) \
                                .groupByKey() \
                                .mapValues(lambda times: (sorted([t for t, _ in times]), len(times)))

# formatting the filtered results to include frequency, ascension, declination, count, and times
formatted_filtered_rdd_n3 = sorted_grouped_rdd_n3.map(lambda x: f"{x[0]}: Count = {x[1][1]}, Times = {x[1][0]}")

total_blips_n3 = filtered_blip_data_rdd_n3.map(lambda x: x[1][1]).sum()
print(f"Total number of blips with 3 sd 3030: {total_blips_n3}")

#this outputs 28

#---------------------------------------------------------------------------------

# making this range shorter for both, since the above range was very huge ~ 10 st dev from mid point
# making this now ~ 1 st dev (so 0.2 units range), this won't be as useful but just to try

# frequency and coordinate ranges
frequency_range_n4 = (4447.9426, 4448.1426) # from output above: median = 4448.04258
ascension_range_n4 = (86.39724, 86.59724) # median: 86.49724
declination_range_n4 = (68.02535, 68.22535) # median: 68.12535

# filtering the blip data based on the specified ranges
filtered_blip_data_rdd_n4 = blip_data.filter(
    lambda x: frequency_range_n4[0] <= x[0][0] <= frequency_range_n4[1] and 
               ascension_range_n4[0] <= x[0][1] <= ascension_range_n4[1] and 
               declination_range_n4[0] <= x[0][2] <= declination_range_n4[1]
)

# flattening the data for sorting
flattened_rdd_n4 = filtered_blip_data_rdd_n4.flatMap(
    lambda x: [((x[0], time), 1) for time in x[1][0]]
)

# sorting the flattened data by time
sorted_rdd_n4 = flattened_rdd_n4.sortByKey(keyfunc=lambda x: x[1])  

# regrouping to collect times and counts by (frequency, ascension, declination) key
sorted_grouped_rdd_n4 = sorted_rdd_n4.map(lambda x: (x[0][0], (x[0][1], 1))) \
                                .groupByKey() \
                                .mapValues(lambda times: (sorted([t for t, _ in times]), len(times)))

# formatting the filtered results to include frequency, ascension, declination, count, and times
formatted_filtered_rdd_n4 = sorted_grouped_rdd_n4.map(lambda x: f"{x[0]}: Count = {x[1][1]}, Times = {x[1][0]}")

total_blips_n4 = filtered_blip_data_rdd_n4.map(lambda x: x[1][1]).sum()
print(f"Total number of blips with 1 sd 4448: {total_blips_n4}")

# this outputs 16

#--------------------------------------------------------------------------

# similarly for the other

frequency_range_n3 = (3030.58861, 3030.78861) # median: 3030.68861
ascension_range_n3 = (104.40624, 104.60624) # median: 104.50624
declination_range_n3 = (111.32846, 111.52846) # median: 111.42846

# filtering the blip data based on the specified ranges
filtered_blip_data_rdd_n3 = blip_data.filter(
    lambda x: frequency_range_n3[0] <= x[0][0] <= frequency_range_n3[1] and 
               ascension_range_n3[0] <= x[0][1] <= ascension_range_n3[1] and 
               declination_range_n3[0] <= x[0][2] <= declination_range_n3[1]
)

# flattening the data for sorting
flattened_rdd_n3 = filtered_blip_data_rdd_n3.flatMap(
    lambda x: [((x[0], time), 1) for time in x[1][0]]
)

# sorting the flattened data by time
sorted_rdd_n3 = flattened_rdd_n3.sortByKey(keyfunc=lambda x: x[1])  

# regrouping to collect times and counts by (frequency, ascension, declination) key
sorted_grouped_rdd_n3 = sorted_rdd_n3.map(lambda x: (x[0][0], (x[0][1], 1))) \
                                .groupByKey() \
                                .mapValues(lambda times: (sorted([t for t, _ in times]), len(times)))

# formatting the filtered results to include frequency, ascension, declination, count, and times
formatted_filtered_rdd_n3 = sorted_grouped_rdd_n3.map(lambda x: f"{x[0]}: Count = {x[1][1]}, Times = {x[1][0]}")

total_blips_n3 = filtered_blip_data_rdd_n3.map(lambda x: x[1][1]).sum()
print(f"Total number of blips with 1 sd 3030: {total_blips_n3}")

#this outputs 16



