class Site:  # Class of site properties. initialize all attributes to None
    def __init__(self):
        self.period = None  # '(-1),(0),(real variable)' period of vibration =-1->PGV; =0->PGA; >0->SA
        self.Rrup = None  # closest distance coseismic rupture (km)
        self.Rjb = None  # closest horizontal distance coseismic rupture (km)
        self.Rx = None  # distance measured perpendicular to fault strike from surface projection of updip edge of the fault rupture (+ve in downdip dir) (km)
        self.Rtvz = None  # source-to-site distance in the Taupo volcanic zone (TVZ) (km)
        self.V30measured = None  # yes =True (i.e. from Vs tests); no=False (i.e. estimated from geology)
        self.V30 = None  # shear wave velocity at 30m depth (m/s)
        self.Z1pt0 = None  # depth to the 1.0km/s shear wave velocity horizon (optional, uses default relationship otherwise
        self.Z_2p5 = None
        self.g = 981.0  # gravity (cm s^-2)
        self.defn = None
        self.orientation = 'average'
        self.Zvs = None


class Fault:  # Class of fault properties. initialize all attributes to None
    def __init__(self):
        self.Mw = None  # moment tensor magnitude
        self.rake = None  # rake angle (degrees)
        self.dip = None  # dip angle (degrees)
        self.Ztor = None  # depth to top of coseismic rupture (km)
        self.rupture_type = None  # Valid values are: N, R, SS and None which correlate to Normal, Reverse, Strike-Slip and Unknown
        self.faultstyle = None