
import sys

sys.path.append('../..')

from GMM_models.zhou_2006 import Zhaoetal_2006_Sa
from classdef import Site, Fault

site = Site()

site.Rjb = 97.24
site.Rrup = 98.36
site.V30 = 500.0
site.siteclass = 'hardsoil'

fault = Fault()
fault.Mw = 7.18
fault.Ztor = 0.0
fault.dip = 35.0
fault.rake = 90.0
fault.rupture_type = 'R'
fault.faultstyle = 'reverse'
fault.h = 15

print Zhaoetal_2006_Sa(site, fault, 0)
