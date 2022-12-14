# -*- coding: utf-8 -*-

from .env import set_workdir
# initiate the .env file
set_workdir()

from .utils import collect_data
from ._version import __version__

from . import spec2Dreduc
from . import spec1Dreduc
from . import wavecalib
from . import fluxcalib
from . import utils