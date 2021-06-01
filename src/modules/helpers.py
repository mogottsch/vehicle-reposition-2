from math import sqrt, sin, cos, atan2, radians


def calc_distance_haversine(coord1, coord2):
    """Returns the distance between two given coordiantes in km.

    Parameters
    ----------
    coord1 : tuple, required
        tuple containing the latitude and longitude of a coordinate.
    coord2 : tuple, required
        tuple containing the latitude and longitude of a coordinate.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}bytes"
