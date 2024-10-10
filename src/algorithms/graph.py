# System Imports
import math

# Library Imports
import pandas as pd

# Project Imports
import utilities.logger as logger

# Constant Variables
LAT_OFFSET = 0.0015
LONG_OFFSET = 0.0013

# Global Variables
df = None


def load_data():
    global df
    # Load in the 'scats_data.csv' file
    file_location = "../training_data/scats_data.csv"

    df = pd.read_csv(file_location)

    # Fix location names.
    df["Location"] = df["Location"].replace(
        {
            "HIGH STREET_RD": "HIGH_STREET_RD",
            "STUDLEY PARK_RD": "STUDLEY_PARK_RD",
            "MONT ALBERT_RD": "MONT_ALBERT_RD",
        },
        regex=True,
    )


def generate_graph():
    global df

    # get unique values of 'Location' column
    locations = df["Location"]

    # get longitude and latitude
    longitudes = df["NB_LONGITUDE"]
    latitudes = df["NB_LATITUDE"]

    # get scats number
    scats_numbers = df["SCATS Number"]

    # Compute a separate dataframe which is all unique rows by Location
    unique_df = df.drop_duplicates(subset=["Location"])

    graph = {}

    for index, scat in enumerate(scats_numbers):
        location_split = locations[index].split(" ")

        longitude = longitudes[index]
        latitude = latitudes[index]

        intersection = int(scat)

        direction = location_split[1]

        opposite_direction = get_opposite_direction(direction)

        search_str = f"{location_split[0]} {opposite_direction}".lower()

        # Search the unique dataframe for a 'Location' that contains the first location and direction
        first_loc_df = unique_df[
            (unique_df["Location"].str.lower().str.contains(search_str))
            & (unique_df["SCATS Number"] != scat)
        ]

        closest_scat = None
        min_distance = float("inf")

        # Find the closest SCAT based on longitude and latitude
        for _, row in first_loc_df.iterrows():
            dist = math.sqrt(
                (row["NB_LONGITUDE"] - longitude) ** 2
                + (row["NB_LATITUDE"] - latitude) ** 2
            )

            if dist < min_distance:
                min_distance = dist
                closest_scat = row["SCATS Number"]

        entry = f"{closest_scat}_{opposite_direction}"

        if closest_scat is not None:
            if graph.get(intersection) is None:
                graph[intersection] = [entry]
            else:
                if entry not in graph[intersection]:
                    graph[intersection].append(entry)

    logger.log(graph)
    logger.log("[+] Graph generated successfully")

    return graph


def get_opposite_direction(direction):

    opposites = {
        "N": "S",
        "S": "N",
        "E": "W",
        "W": "E",
        "NE": "SW",
        "SW": "NE",
        "NW": "SE",
        "SE": "NW",
    }

    return opposites.get(direction, None)


# Get all SCAT numbers
def get_all_scats():
    global df

    return df["SCATS Number"].unique()


def get_coords_by_scat(scat_number):
    global df

    row = df[df["SCATS Number"] == scat_number]
    latitude = row["NB_LATITUDE"].values[0] + LAT_OFFSET
    longitude = row["NB_LONGITUDE"].values[0] + LONG_OFFSET

    return latitude, longitude


def calculate_distance(start, end):
    start_lat, start_long = get_coords_by_scat(start)
    end_lat, end_long = get_coords_by_scat(end)

    # 0.01 of a degree is 1km
    # degree difference x 100
    a = abs(start_lat - end_lat) * 100
    b = abs(start_long - end_long) * 100

    # c^2 = a^2 +b^2
    c = math.sqrt(a**2 + b**2)

    return c
