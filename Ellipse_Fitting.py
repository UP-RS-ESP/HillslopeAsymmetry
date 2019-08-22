'''
Functions to fit an ellipse to a set of slope-aspect points based on a DEM

Author: Taylor Smith
v1.0, Aug 2019
'''

import numpy as np
from numpy.linalg import eig, inv, svd
import math, os, time, subprocess
import gdal, gdalnumeric, ogr, osr

#Decide on the number of aspect bins to fit an ellipse over
step = 1
asp_bins = np.arange(0,360 + step,step)
slice_width = (2*np.pi) / (asp_bins.shape[0] - 1) #Define the width of the slices
polar_bins = ((asp_bins-1)).astype(float) * (np.pi/180) #Get the bin coordinates in radians
FNULL = open(os.devnull, 'w')

#Fitting
def fitEllipse(x,y):
    #http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2
    C[1,1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_axis_length( a ):
    #http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation2( a ):
    #http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))

def ellipse_angle_of_rotation( a ):
    #http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    # ax^2+2bxy+cy^2+2dx+2fy+g=0 
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2
        
def ellipse_center(a):
    #http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def raw_comp(slp, asp):
    '''Function to get the median slope in cardinal aspect bins for comparison
    e.g., after Poulos et al., 2012'''
    
    idxN = np.logical_or(asp < 45, asp > 315)
    idxS = np.logical_and(asp > 135, asp < 225)
    idxE = np.logical_and(asp > 45, asp < 135)
    idxW = np.logical_and(asp > 225, asp < 315)
    
    def data_pull(slp, idx):
        slp_sub = slp[idx]
        return [np.nanmean(slp_sub), np.nanmedian(slp_sub), np.nanstd(slp_sub)]
    
    SN = data_pull(slp, idxN)
    SS = data_pull(slp, idxS)
    SE = data_pull(slp, idxE)
    SW = data_pull(slp, idxW)
    
    return SN, SS, SE, SW     

def build_bad_idx(asp):
    '''Create an index to remove points directly at cardinal directions'''
    a = np.logical_and(asp >= 0, asp <= 1)
    b = np.logical_and(asp >= 44, asp <= 46)
    c = np.logical_and(asp >= 89, asp <= 91)
    d = np.logical_and(asp >= 134, asp <= 136)
    e = np.logical_and(asp >= 179, asp <= 181)
    f = np.logical_and(asp >= 224, asp <= 226)
    g = np.logical_and(asp >= 269, asp <= 271)
    h = np.logical_and(asp >= 314, asp <= 316)
    i = np.logical_and(asp >= 359, asp <= 360)
    
    idx = np.logical_or.reduce((a, b, c, d, e, f, g, h,i))
    return np.where(idx == True)
    
def cardinal_offsets(ell):
    '''Get the N-S and E-W ellipse offsets'''
    try:
        (cx, cy) = ellipse_center(ell)
        e, w = cy, cy * -1
        n, s = cx, cx * -1
        
        return n, s, e, w
    except:
        return np.nan, np.nan, np.nan, np.nan

def ell_meta(ell):
    '''
    Store metadata from a given ellipse
    Returns [minor axis, major axis, eccentricity, north steepening, east steepening]
    '''
    try:
        axes = ellipse_axis_length(ell)
        a, b = np.nanmax(axes), np.nanmin(axes)
        e = np.sqrt(1-((b**2)/(a**2)))
        north, s, east, w = cardinal_offsets(ell)
    except:
        a, b, e, north, east = (np.nan,)*5
    
    metadata = [a, b, e, north, east]
    return metadata

def fit_ellipse(slp, asp):
    #Generate the slope-aspect distribution
    out = np.empty(asp_bins.shape[0])
    out.fill(np.nan)
    for i, ma in enumerate(asp_bins[1:]):
        mi = asp_bins[i]
        ss = slp[np.logical_and(asp < ma, asp >= mi)]
        avslp = np.nanmedian(ss)
        out[i] = avslp
    del slp, asp
        
    sub_bins = np.empty(asp_bins.shape[0])
    sub_bins[:] = polar_bins.copy()
    
    #Remove cardinal directions
    badidx = build_bad_idx(asp_bins)
    out[badidx] = np.nan
    
    #Remove NaNs
    sub_bins = sub_bins[~np.isnan(out)]
    out = out[~np.isnan(out)]
            
    #Convert to polar coords
    xcoord = np.cos(sub_bins) * out
    ycoord = np.sin(sub_bins) * out
    
    #Remove NaNs
    xcoord = xcoord[~np.isnan(ycoord)]
    ycoord = ycoord[~np.isnan(ycoord)]

    #Do the ellipse fitting
    try:
        ell = fitEllipse(xcoord, ycoord)
    except:
        ell = np.nan
        
    #Save and return the original slope-aspect distribution
    out_raw = np.squeeze(np.dstack((out, sub_bins)))
    
    return ell, out_raw

#Subsetting by Shapefile
def choose_utm_zone(feat):
    ''' Generate an EPSG code for UTM projection for a given lat/lon '''
    lon = feat.geometry().Centroid().GetX()
    lat = feat.geometry().Centroid().GetY()
    
    #https://gis.stackexchange.com/questions/269518/auto-select-suitable-utm-zone-based-on-grid-intersection
    utm_band = str(int(math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    
    return int(float(epsg_code))

def clip_and_proj(feature, tiffile):
    #Get UTM zone for reprojection
    epsg = choose_utm_zone(feature)
    
    cs_in = cs_base.ExportToWkt()
    cs_out = osr.SpatialReference()
    cs_out.ImportFromEPSG(epsg)
    cs_out = cs_out.ExportToWkt()
    
    #Create a temporary clipping shape
    randint = str(np.random.randint(0,1e6))
    geom = feature.geometry()
    tmp = tmpdir + randint + 'tmp.shp'
    drv = ogr.GetDriverByName('ESRI Shapefile')
    dst_ds = drv.CreateDataSource(tmp)
    dst_layer = dst_ds.CreateLayer('Clipper', srs=cs_base)
    f = ogr.Feature(dst_layer.GetLayerDefn())
    f.SetGeometry(geom)
    f.SetFID(0)
    dst_layer.CreateFeature(f)
    dst_ds.Destroy()
    f = drv = dst_ds = dst_layer = geom = None
    
    #Get nodata values
    c = gdal.Open(tiffile)
    b = c.GetRasterBand(1)
    if not b.GetNoDataValue() == None:
        ndataval = str(b.GetNoDataValue())
    else:
        if b.GetMinimum() == -9999:
            ndataval = '-9999'
        else:
            ndataval = 'nan'
    c = b = None
    
    #Clip and reproject the base data
    outfile = tmpdir + str(np.random.randint(0,1e6)) + '_reproj_tmp.tif'
    gdal_comm = ['gdalwarp', '-overwrite', '-s_srs', cs_in, '-t_srs', cs_out, '-r', 'bilinear', '-cutline', tmp, '-crop_to_cutline', '-srcnodata', ndataval, tiffile, outfile]
    #print gdal_comm
    process = subprocess.Popen(gdal_comm, stdout=subprocess.PIPE, stderr=FNULL)
    process.communicate()
    exit_code = process.wait()
    
    #Clean up temp shapefile
    os.remove(tmp)
    os.remove(tmpdir + randint + 'tmp.shx')
    os.remove(tmpdir + randint + 'tmp.prj')
    os.remove(tmpdir + randint + 'tmp.dbf')
            
    return outfile
    
def slp_asp_to_ell(DEM):
    '''Fit ellipse against a given clipped DEM'''
    #Generate slope with GDAL
    slp = tmpdir + str(np.random.randint(0,1e6)) + '_reproj_slope.tif'
    gdal_comm = ['gdaldem', 'Slope', '-compute_edges', DEM, slp]
    process = subprocess.Popen(gdal_comm, stdout=subprocess.PIPE, stderr=FNULL)
    process.communicate()
    _ = process.wait()
    if not os.path.exists(slp):
        time.sleep(1)
    
    #Generate aspect with GDAL
    asp = tmpdir + str(np.random.randint(0,1e6)) + '_reproj_aspect.tif'
    gdal_comm = ['gdaldem', 'Aspect', '-compute_edges', DEM, asp]
    process = subprocess.Popen(gdal_comm, stdout=subprocess.PIPE, stderr=FNULL)
    process.communicate()
    _ = process.wait()
    if not os.path.exists(asp):
        time.sleep(1)
        
    #Load in the data and mask out NaNs
    slope = np.array(gdalnumeric.LoadFile(slp).astype(float))
    slope[slope < 0] = np.nan

    aspect = np.array(gdalnumeric.LoadFile(asp).astype(float))
    aspect[aspect < 0] = np.nan 
    
    ell, out_raw = fit_ellipse(slope, aspect)
    
    #Pull the center, eccentricity, and north/east steepening from the ellipse
    metadata = ell_meta(ell)
    
    return ell, out_raw, metadata

def Process_Feature(i):
    #Set the random seed so names of temp files don't overlap
    np.random.seed(i)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shape_fid, 0)
    layer = dataSource.GetLayer()
    feat = layer.GetFeature(i)
    
    DEM = clip_and_proj(feat, base_DEM)
    ell, out_raw, metadata = slp_asp_to_ell(DEM)
    
    return ell, out_raw, metadata

#Simple plotting utilities
def signed_dists(center, phi, a, b, x, y, xx, yy):
    in_out = ((np.cos(phi)*(x - center[0]) + np.sin(phi)*(y - center[1]))**2)/(a**2) + ((np.sin(phi)*(x - center[0]) + np.cos(phi)*(y-center[1]))**2)/(b**2)
    if in_out <= 1:
        #Neg value, inside Ell
        s = -1
    elif in_out > 1:
        #Pos value, outside Ell
        s = 1
    dist = s * np.sqrt((x - xx)**2 + (y - yy)**2)
    return dist
    
def pick_phi(ell, out):
    '''Test whether ellipse is rotated 90-degrees to the data
    NOTE: This is a bug from the SVM method -- sometimes the fits are rotated 90 degrees. 
    This function makes sure that the proper ellipse fit is chosen. This only effects the 
    chose of semimajor/semiminor axes and rotation of ellipse -- does not impact the centroid
    or eccentricity of the ellipse fit.
    '''
    center = ellipse_center(ell)
    phi = ellipse_angle_of_rotation(ell)
    axes = ellipse_axis_length(ell)
    #R = np.arange(0,2*np.pi, 0.001)
    R = out[:,1]
    a, b = np.nanmax(axes), np.nanmin(axes)
    xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
    yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
    
    phi2 = phi + np.radians(90)
    xx_ = center[0] + a*np.cos(R)*np.cos(phi2) - b*np.sin(R)*np.sin(phi2)
    yy_ = center[1] + a*np.cos(R)*np.sin(phi2) + b*np.sin(R)*np.cos(phi2)
    
    bins = out[:,1]
    out_vals = out[:,0]
    xcoord = np.cos(bins) * out_vals
    ycoord = np.sin(bins) * out_vals
    xcoord = xcoord[~np.isnan(ycoord)]
    ycoord = ycoord[~np.isnan(ycoord)]
        
    dists = np.sqrt((xcoord - xx)**2 + (ycoord - yy)**2)
    MAE_phi = np.nanmean(np.abs(dists))
    
    dists = np.sqrt((xcoord - xx_)**2 + (ycoord - yy_)**2)
    MAE_phi2 = np.nanmean(np.abs(dists))
    
    if MAE_phi <= MAE_phi2:
        phi_out, MAE_out = phi, MAE_phi
        xx, yy = xx, yy
    elif MAE_phi2 < MAE_phi:
        phi_out, MAE_out = phi2, MAE_phi2
        xx, yy = xx_, yy_
    
    dists = []
    for i in range(xcoord.shape[0]):
        dists.append(signed_dists(center, phi, a, b, xcoord[i], ycoord[i], xx[i], yy[i]))
        
    mn_dist = np.nanmedian(dists)
    sum_dist = np.nansum(dists)
    
    return phi_out, MAE_out, mn_dist, sum_dist

def plot_ellipse(ax, ell_data):  
    ell, out, metadata = ell_data     
    
    bins = out[:,1]
    out_vals = out[:,0]
    
    xcoord = np.cos(bins) * out_vals
    ycoord = np.sin(bins) * out_vals
    xcoord = xcoord[~np.isnan(ycoord)]
    ycoord = ycoord[~np.isnan(ycoord)]
    
    ax.plot(xcoord, ycoord, 'r.')
    
    center = ellipse_center(ell)
    axes = ellipse_axis_length(ell)
    
    #Get the 'correct' phi -- sometimes the angle is off by 90 degrees. Also pull fitting error
    phi, MAE, md_dist, sum_dist = pick_phi(ell, out)
    #phi = ellipse_angle_of_rotation(ell)
    
    arc = 2
    R = np.arange(0,arc*np.pi, 0.001)
    a, b = np.nanmax(axes), np.nanmin(axes)
    
    av_rad = np.nanmean([a, b])
    mae_pct = (MAE / av_rad) * 100
    md_pct = np.abs(md_dist / av_rad) * 100
    e = np.sqrt(1-((b**2)/(a**2)))    

    xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
    yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
     
    ax.plot(xx,yy, 'k--', lw=2)
    ya, xa = center[1] + a*np.sin(phi), center[0] + a*np.cos(phi)
    ya2, xa2 = center[1] - a*np.sin(phi), center[0] - a*np.cos(phi)
    ax.plot((xa, xa2),(ya,ya2), 'k--')
    
    phi2 = phi + np.radians(90)
    ya, xa = center[1] + b*np.sin(phi2), center[0] + b*np.cos(phi2)
    ya2, xa2 = center[1] - b*np.sin(phi2), center[0] - b*np.cos(phi2)
    ax.plot((xa, xa2),(ya,ya2), 'k--')

    if phi < 0:
        phi = phi + 2*np.pi
    
    ax.plot(center[0], center[1], 'bo')
    ax.grid('on')
    ax.axis('equal')
    tit = 'Eccentricity: %.3g\n MD: %.3g, MD Pct: %.3g, MAE: %.3g, MAE Pct: %.3g\n Center X: %.3g, Center Y: %.3g' % (e, md_dist, md_pct, MAE, mae_pct, center[0], center[1])
    ax.set_title(tit)

#Main functions
tmpdir = #Directory to hold temporary files
shape_fid = #File to clip from (e.g., a set of watersheds, bounding boxes, etc)
base_DEM = #DEM covering whole study area           
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(shape_fid, 0)
layer = dataSource.GetLayer()
cs_base = layer.GetSpatialRef()
numel = layer.GetFeatureCount()
del layer, dataSource, driver

ell_dict = {}
for i in range(numel):
    ell, out_raw, metadata = Process_Feature(i)
    ell_dict[i] = [ell, out_raw, metadata]
    
import matplotlib.pyplot as plt
plt.close('all')
f, ax = plt.subplots(1)
plot_ellipse(ax, ell_dict[0])
