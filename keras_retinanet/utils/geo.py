import numpy as np
import math

from snappy import PixelPos

def utmToLatLng(zone, easting, northing, northernHemisphere=True):
    if not northernHemisphere:
        northing = 10000000 - northing

    a = 6378137
    e = 0.081819191
    e1sq = 0.006739497
    k0 = 0.9996

    arc = northing / k0
    mu = arc / (a * (1 - math.pow(e, 2) / 4.0 - 3 * math.pow(e, 4) / 64.0 - 5 * math.pow(e, 6) / 256.0))

    ei = (1 - math.pow((1 - e * e), (1 / 2.0))) / (1 + math.pow((1 - e * e), (1 / 2.0)))

    ca = 3 * ei / 2 - 27 * math.pow(ei, 3) / 32.0

    cb = 21 * math.pow(ei, 2) / 16 - 55 * math.pow(ei, 4) / 32
    cc = 151 * math.pow(ei, 3) / 96
    cd = 1097 * math.pow(ei, 4) / 512
    phi1 = mu + ca * math.sin(2 * mu) + cb * math.sin(4 * mu) + cc * math.sin(6 * mu) + cd * math.sin(8 * mu)

    n0 = a / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (1 / 2.0))

    r0 = a * (1 - e * e) / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (3 / 2.0))
    fact1 = n0 * math.tan(phi1) / r0

    _a1 = 500000 - easting
    dd0 = _a1 / (n0 * k0)
    fact2 = dd0 * dd0 / 2

    t0 = math.pow(math.tan(phi1), 2)
    Q0 = e1sq * math.pow(math.cos(phi1), 2)
    fact3 = (5 + 3 * t0 + 10 * Q0 - 4 * Q0 * Q0 - 9 * e1sq) * math.pow(dd0, 4) / 24

    fact4 = (61 + 90 * t0 + 298 * Q0 + 45 * t0 * t0 - 252 * e1sq - 3 * Q0 * Q0) * math.pow(dd0, 6) / 720

    lof1 = _a1 / (n0 * k0)
    lof2 = (1 + 2 * t0 + Q0) * math.pow(dd0, 3) / 6.0
    lof3 = (5 - 2 * Q0 + 28 * t0 - 3 * math.pow(Q0, 2) + 8 * e1sq + 24 * math.pow(t0, 2)) * math.pow(dd0, 5) / 120
    _a2 = (lof1 - lof2 + lof3) / math.cos(phi1)
    _a3 = _a2 * 180 / math.pi

    latitude = 180 * (phi1 - fact1 * (fact2 + fact3 + fact4)) / math.pi

    if not northernHemisphere:
        latitude = -latitude

    longitude = ((zone > 0) and (6 * zone - 183.0) or 3.0) - _a3

    return (latitude, longitude)
    
def xyToLatLonTiff(dataset, x, y):
    ulx, xres, xskew, uly, yskew, yres  = dataset.GetGeoTransform()
    
    pos_x = ulx + (x * xres)
    pos_y = uly + (y * yres)
    
    return pos_x, pos_y

def readTiffTile(dataset, xLeft, yTop, sizeX, sizeY, size_band, scale_factor=1.0):
    data = np.zeros((int(sizeY * scale_factor), int(sizeX * scale_factor), size_band), dtype=np.float)
    for i in range(size_band):
        band_data = dataset.GetRasterBand(i + 1).ReadAsArray(xLeft, yTop, sizeX, sizeY)

        data[..., i] = np.resize(band_data, (int(sizeY * scale_factor), int(sizeX * scale_factor)))
    
    return data

def readDimTile(dataset, xLeft, yTop, sizeX, sizeY, size_band, scale_factor=1.0):
    bandName    = dataset.getBandNames()

    data = np.zeros((int(sizeY * scale_factor), int(sizeX * scale_factor), size_band), dtype=np.float)
    for i in range(size_band):
        band_data = np.zeros(sizeX * sizeY)
        dataset.getBand(bandName[i]).readPixels(xLeft, yTop, sizeX, sizeY, band_data)
        band_data = band_data.reshape(sizeY, sizeX)

        data[..., i] = np.resize(band_data, (int(sizeY * scale_factor), int(sizeX * scale_factor)))
    
    return data

def xyToLatLonDim(dataset, x, y):
    pos = dataset.getSceneGeoCoding().getGeoPos(PixelPos(x, y), None)
    return pos.getLon(), pos.getLat()