import zipfile as z
from re import sub

# WWV10 Nodes of interest
# N0000008 - S1 - EN91fl - WWV10
# --> CALLSIGN AD8Y

# N0000013 - S1 - DN70ln - WWV10

years = [i for i in range(2019, 2024)]
years = [str(i) for i in years]
years = ['2021']

for year in years:
    with z.ZipFile('D:/Sabastian/Perry_Lab/GrapeV1Data_8-23/' + year + '.zip', 'r') as zip:
        # printing all the contents of the zip file
        # zip.printdir()
        names = zip.namelist()

    # print(names)
    names.sort(key=lambda f: int(sub('\D', '', f)))

    dates = []
    nodes = []
    radioIDs = []
    gridsqrs = []
    beacons = []

    unique_beacons = []

    for name in names:
        namesplit = str(name).split('_')

        date = namesplit[0].split('T')[0]
        node = namesplit[1]
        radioID = namesplit[2]
        gridsqr = namesplit[3]
        beacon = namesplit[5].split('.')[0]

        if node not in nodes:
            if beacon == 'WWV10':
                unique_beacons.append(node + ' - ' + radioID + ' - ' + gridsqr + ' - ' + beacon)

        dates.append(date)
        nodes.append(node)
        radioIDs.append(radioID)
        gridsqrs.append(gridsqr)
        beacons.append(beacon)


    # Writes the zip data to a text file
    with open(year + '.txt', 'w') as f:

        f.write('Node - RadioID - Gridsqr - Beacon \n'
                '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n\n')

        for i in range(0, len(unique_beacons)):
            # f.write(names[i])
            # f.write(dates[i])
            # f.write(nodes[i] + ' - ')
            # f.write(radioIDs[i] + ' - ')
            # f.write(gridsqrs[i] + ' - ')
            # f.write(beacons[i])
            f.write(unique_beacons[i])

            f.write('\n\n')
