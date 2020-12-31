# Copyright (C) 2020 Zach (Yuzhe) Ni 
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
#
# Standard Library Imports
#

import numpy as np
import collections
import random
from tensorflow.keras.initializers import Identity
import tensorflow.keras.layers as L
import tensorflow.keras as K
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from collections import deque
import random
import math
