#numpy_to_torch_n_back

import numpy as np
a = np.random.rand(4,3)
a

b = torch.from_numpy(a)
b

b.numpy()



#The memory is shared between the Numpy array and Torch tensor, so if you change the values in-place of one object, the other will change as well.


# Multiply PyTorch Tensor by 2, in place
b.mul_(2)


# Numpy array matches new values from Tensor
a
