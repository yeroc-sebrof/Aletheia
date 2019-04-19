"""
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
File type is added to one list
Header and Footer patterns are added to another list

The list above is then parsed and all entries are converted to Hex values only

We will then put these Hex patterns in their own file
"""
import os
import sys
import re
import time
import wget

# https://raw.githubusercontent.com/sleuthkit/scalpel/master/scalpel.conf

debug = True

if (os.path.isfile("aletheia.conf")) & (not debug):
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

currentLine = []  # Holder for each line after the split

patternTypes = []   # Holds the File types
patterns = []       # Holds the patterns as we go for the 'simplified' config

# read each line of the file and act appropriately
for line in existingConf:
    if len(line) == 0:
        continue

    if line[0] == '#':
        continue

    currentLine = line.split()

    if len(currentLine) > 4:
        patternTypes.append(currentLine[0])
        patterns.append(currentLine[3])
        patterns.append(currentLine[4])

    continue

print("There is %i patterns" % len(patterns))

# Patterns successfully collected
existingConf.close()
del currentLine

# Pre-req for searching
halfWayHexSearch = re.compile('(?<!\\\\)x')
areYouHexSearch = re.compile('(\\\\x[0-9a-f]{2})+')  # Matches with >= 1 Hex character


def halfway_hex_conv(not_quite_hex):
    response = halfWayHexSearch.search(not_quite_hex)

    while response:
        not_quite_hex = not_quite_hex[:response.span()[0]] + "\\" + not_quite_hex[response.span()[0]:]

        response = halfWayHexSearch.search(not_quite_hex)

    return not_quite_hex


def all_hex_pls(not_all_hex):
    return


print("Starting to manage each pattern")

for x in range(len(patterns)):
    # If the pattern has a wildcard we can't use it
    if '?' in patterns[x]:  # Any real ? would be in Hex
        if debug:
            print("Pattern contains a ? -- %s" % patterns[x])
            print(patterns[x].replace('?', ''), end="\n\n")
        patterns[x] = patterns[x].replace('?', '')

    # If the pattern has hex that only starts with the x delimiter over the \x
    if halfWayHexSearch.search(patterns[x]):
        if debug:
            print("Halfway Hex fixing the following pattern - %s" % patterns[x])
        patterns[x] = halfway_hex_conv(patterns[x])
        if debug:
            print("Done - %s" % patterns[x], end="\n\n")

    isItHex = areYouHexSearch.search(patterns[x])
    if isItHex:
        if (isItHex.span()[0] == 0) & (isItHex.span()[1] == len(patterns[x])):
            if debug:
                print("All Hex? %s" % patterns[x], end='\n\n')
        else:  # Some Hex
            if debug:
                print("This was a little bit Hex %s" % patterns[x], end='')

            letsMakeThisHex = ""
            tempStorage = ""

            while len(patterns[x]) > 0:
                if isItHex.span()[0] != 0:
                    tempStorage = patterns[x][:isItHex.span()[0]]
                    for cheekyChar in tempStorage:
                        letsMakeThisHex += "\\" + hex(ord(cheekyChar))[1:]
                    patterns[x] = patterns[x][isItHex.span()[0]:]
                else:
                    letsMakeThisHex += patterns[x][:isItHex.span()[1]]
                    patterns[x] = patterns[x][isItHex.span()[1]:]

                isItHex = areYouHexSearch.search(patterns[x])

            patterns[x] = letsMakeThisHex
            if debug:
                print(" now it is ", end='')
                print(patterns[x], end='\n\n')

    else:  # Zero Hex
        if debug:
            print("No Hex -- Before: %s" % patterns[x], end='')
        letsMakeThisHex = ""

        for cheekyChar in patterns[x]:
            letsMakeThisHex += '\\' + hex(ord(cheekyChar))[1:]

        patterns[x] = letsMakeThisHex
        if debug:
            print(" -- After: %s" % patterns[x], end='\n\n')


if debug:
    print("The following output is the list")
    print(patterns)
    exit(0)

# Open new config file to write to
try:
    existingConf = open("aletheia.conf", "xt+")
except FileExistsError:
    existingConf = open("aletheia.conf", "wt")  # This should completely write over the file

# write to the file in a CSV style format: 'Type, Head, Foot\n'; Where Head and Foot store the Hex Values
for x in range(len(patternTypes)):
    existingConf.write(patternTypes[x])  # Type
    existingConf.write(", %s" % patterns[x*2])  # Head
    existingConf.write(", %s\n" % patterns[(x*2)+1])  # Foot
