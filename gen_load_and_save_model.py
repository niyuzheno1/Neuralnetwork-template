  
# Copyright (C) 2020 Zach (Yuzhe) Ni 
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
#
# Save and load models including numpy and tensorflow 
#

def save(m,name, isnumpy=False):
  if isnumpy == False:
    m.save(name)
  else:
    np.savetxt(name,m.flatten())

def load(name, isnumpy=False, dim=None):
  if isnumpy == False:
    return load_model(name)
  else:
    m = np.loadtxt(name)
    if dim == None:
      return m
    else:
      return m.reshape(dim)
