#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 START_YEAR END_YEAR"
    exit 1
fi

START_YEAR=$1
END_YEAR=$2

# Validate year arguments
if ! [[ "$START_YEAR" =~ ^[0-9]{4}$ ]] || ! [[ "$END_YEAR" =~ ^[0-9]{4}$ ]]; then
    echo "Error: Years must be 4-digit numbers"
    exit 1
fi

if [ "$START_YEAR" -gt "$END_YEAR" ]; then
    echo "Error: START_YEAR must be less than or equal to END_YEAR"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p backend/data/raw

# Download files for each year in range
for Y in $(seq $START_YEAR $END_YEAR); do
    OUTFILE="backend/data/raw/statcast_${Y}.csv"
    if [ ! -f "$OUTFILE" ]; then
        echo "Downloading Statcast data for $Y..."
        # Add a 5-second delay between requests
        sleep 5
        # Use -v for verbose output and -f to fail on HTTP errors
        if curl -v -f -L -o "$OUTFILE" "https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7CPO%7CS%7C=&hfSea=&hfSit=&player_type=pitcher&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfInfield=&team=&position=&hfOutfield=&hfRO=&home_road=&hfFlag=&hfPull=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&min_abs=0&type=details&player_id=&year=${Y}"; then
            # Check if file is valid (not empty and not an error message)
            if [ -s "$OUTFILE" ] && ! grep -q "error" "$OUTFILE"; then
                echo "Successfully downloaded data for $Y"
            else
                echo "Error: Downloaded file for $Y appears to be invalid"
                rm "$OUTFILE"
            fi
        else
            echo "Error: Failed to download data for $Y"
            rm "$OUTFILE" 2>/dev/null
        fi
    else
        echo "File already exists for $Y, skipping..."
    fi
done

exit 0 