# Import packages
import zipfile
from csv import reader
import io

# Define Year Range
years = [i for i in range(2019, 2024)]

nodestrings = []
unodes = []
# Loop through all zipfiles
for year in years:
    unames = []

    print(f'{year}: \n')

    zippath = 'D:/Sabastian/Perry_Lab/GrapeV1Data_8-23'
    zipname = f'{year}.zip'
    archive = zipfile.ZipFile(f'{zippath}/{zipname}', 'r')

    names = archive.namelist()

    # Find Unique nodes and names
    for name in names:
        namesplit = name.split('_')
        node = namesplit[1]

        if node not in unodes:
            # If node is unique, add node and name to ledgers
            unodes.append(node)
            unames.append(name)

    # Extract Lat, Lon from unique nodes
    for uname in unames:
        open_archive = archive.open(uname)

        with io.TextIOWrapper(open_archive, encoding='utf-8') as text_file:
            dataReader = reader(text_file)
            lines = list(dataReader)
            info = lines[0]

            node = info[2]
            lat = info[4]
            lon = info[5]

            nodestring = f'{node}, {lat}, {lon}'
            print(nodestring)

            nodestrings.append(nodestring)

    print('\n')


# Writes the zip data to a text file
with open('node_latlons.txt', 'w') as f:

    f.write('Node - Latitude - Longitude \n'
            '~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n\n')

    for i in range(0, len(nodestrings)):
        f.write(nodestrings[i])
        f.write('\n')

    f.close()
