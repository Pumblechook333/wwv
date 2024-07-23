from math import sin, cos, sqrt, atan2, radians as rad, degrees as deg

# WWV Coordinates
w_lat = 40.67583063
w_lon = -105.038933178

# Newark Coordinates
n_lat = 40.742018
n_lon = -74.178975


def mpt_coords(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = rad(lat1), rad(lon1), rad(lat2), rad(lon2)

    # 1 = origin (WWV), 2 = receiver (Newark)
    X = cos(lat2) * cos(lon2 - lon1)
    Y = cos(lat2) * sin(lon2 - lon1)

    # midpoint (bounce point) latitude
    mlat = atan2(sin(lat1) + sin(lat2), sqrt((cos(lat1) + X) ** 2 + Y ** 2))
    # midpoint (bounce point) longitude
    mlon = lon1 + atan2(Y, cos(lat1) + X)

    mlat, mlon = deg(mlat), deg(mlon)

    return mlat, mlon


def avg_mpt(lat1, lon1, lat2, lon2):
    mlat = (lat1 + lat2) / 2
    mlon = (lon1 + lon2) / 2

    return mlat, mlon


mlat1, mlon1 = mpt_coords(w_lat, w_lon, n_lat, n_lon)

mlat2, mlon2 = avg_mpt(w_lat, w_lon, n_lat, n_lon)

print('Old Midpoint:\n %f, %f\n' % (mlat2, mlon2))

print('New midpoint:\n %f, %f\n' % (mlat1, mlon1))
