def bottom_minus_surface_elevation(ins,outs):
    """
    Used for CReSIS Radar Depth Sounder (RDS) data.
    Calculate actual ice bottom height referenced to WGS84 Ellipsoid.
    See https://data.cresis.ku.edu/data/rds/rds_readme.pdf for more info.
    """
    zb = ins['BOTTOM']  #range to ice bottom (from sensor)
    zs = ins['ELEVATION']  #range to ice surface (from sensor)
    outs['BOTTOM'] = zs - zb  #actual ice bottom height is Elevation minus Bottom
    return True