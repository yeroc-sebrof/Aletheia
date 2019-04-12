'''
This script is used for the conversion of config files.
To save the full Aletheia program from needlessly parsing verbose config files we shall
    simplify Scalpels config with this script

Breakdown
This script will look for if an existing alethia.conf file exists before starting.
If one exists it will ask the user if they wish to continue and expects a yes or y response

To create the config this script will first parse the existing scalpel.conf
If it does not exist an attempt will be made to wget the file.
If internet is not found we exit
If we have the script we open it

In parsing the existing scalpel config we look for two things.
    #'s at the first char and lines of length 0
If neither of these are found we should be on a legitimate pattern line
Header and Footer patterns are then added to a list
No other information is required at this time

We will then put these patterns in their own file

'''
import os
import sys
import time
import wget
# https://raw.githubusercontent.com/sleuthkit/scalpel/master/scalpel.conf

debug = False

if os.path.isfile("aletheia.conf"):
    if input("There already exists a config file.\nDo you wish to continue?").upper()[0] != "Y":
        print("\nYou specified you do not wish to continue, exiting")
        exit(1)

# Open existing config file to read
try:
    existingConf = open("scalpel.conf", "rt")
except FileNotFoundError:
    print("Couldn't fine scalpel.conf, attempting to grab a copy\n")

    try:
        wget.download("https://raw.githubusercontent.com/sleuthkit/scalpel/master/scalpel.conf")
        time.sleep(0.5)
    except:
        print("Unexpected Error, %s\nExiting" % sys.exc_info()[0])
        exit(2)

    print("Success\n")
    existingConf = open("scalpel.conf", "rt")

currentLine = []    # Holder for each line after the split

patterns = []       # Holds the patterns as we go for the 'simplified' config

# read each line of the file and act appropriately
for line in existingConf:
    if len(line) == 0:
        continue

    if line[0] == '#':
        continue

    currentLine = line.split()

    if len(currentLine) > 4:
        patterns.append(currentLine[0])
        patterns.append(currentLine[3])
        patterns.append(currentLine[4])

    continue

print("There is %i patterns" % len(patterns))

existingConf.close()


if debug:
    print("The following output is the list")
    print(patterns)

# Doing this part manually
# for item in patterns:
#   Check the format just shows the hex values of all chars
#   Removing WildCards

# Open new config file to write to
try:
    existingConf = open("aletheia.conf", "xt+")
except FileExistsError:
    existingConf = open("aletheia.conf", "wt") # This should completely write over the file

# write to the file in a CSV style format: 'Type, Head, Foot\n'; Where Head and Foot store the Hex Values
for x in range(0, len(patterns), 3):
    existingConf.write(patterns[x])  # Type
    existingConf.write(", %s" % patterns[x+1])  # Head
    existingConf.write(", %s\n" % patterns[x+2])  # Foot

