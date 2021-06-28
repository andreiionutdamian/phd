# -*- coding: utf-8 -*-
"""
Copyright 2019-2021 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.
* NOTICE:  All information contained herein is, and remains
* the property of Knowledge Investment Group SRL.
* The intellectual and technical concepts contained
* herein are proprietary to Knowledge Investment Group SRL
* and may be covered by Romanian and Foreign Patents,
* patents in process, and are protected by trade secret or copyright law.
* Dissemination of this information or reproduction of this material
* is strictly forbidden unless prior written permission is obtained
* from Knowledge Investment Group SRL.

@copyright: Lummetry.AI
@author: Lummetry.AI - AID
@project:
@description:

"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
  x = np.linspace(0, 1, num=100)
  ce = -np.log(x)
  fl05 = -((1-x)**0.5)*np.log(x)
  fl1 = -((1-x)**1)*np.log(x)
  fl2 = -((1-x)**2)*np.log(x)
  fl5 = -((1-x)**5)*np.log(x)
  
  plt.figure(figsize=(13,8))
  plt.plot(x, ce, label='CE')
  plt.plot(x, fl05, label='FL ɤ=0.5')
  plt.plot(x, fl1, label='FL ɤ=1')
  plt.plot(x, fl2, label='FL ɤ=2')
  plt.plot(x, fl5, label='FL ɤ=5')
  plt.legend(prop={'size': 16})
  plt.show()
  