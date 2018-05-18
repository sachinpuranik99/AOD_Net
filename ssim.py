from keras_contrib.losses import DSSIMObjective
from keras import backend as K

import numpy as np
from keras_contrib.losses import DSSIMObjective
from keras import backend as K

#Shape should be (batch,x,y,channels)
imga = np.random.normal(size=(1,256,256,3))
imgb = np.random.normal(size=(1,256,256,3))

loss_func = DSSIMObjective()

resulting_loss1 = K.eval(loss_func(K.variable(imga),K.variable(imgb)))
resulting_loss2 = K.eval(loss_func(K.variable(imga),K.variable(imga)))

print ("Loss for different images: %.2f" % resulting_loss1)
print ("Loss for same image: %.2f" % resulting_loss2)
