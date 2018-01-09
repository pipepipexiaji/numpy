n = np.zeros(z.shape, dtype=int)
n[1:-1, 1:-1] += (z[ :-2, :-2] + z[ :-2,1:-1] + z[ :-2,2:] +
                            z[1:-1, :-2]                + z[1:-1,2:] +
                             z[2:  , :-2] + z[2:  ,1:-1] + z[2:  ,2:])
# Flatten arrays
n_ =n.ravel()
z_ =z.ravel()
# Apply Rules
R1=np.argwhere((z_ == 1) &(n_ < 2))
R2=np.argwhere((z_ == 1) &(n_ > 3))
R3=np.argwhere( (z_ == 1) & ( (n_== 3) | (n_==2) ) )
R4=np.argwhere( (z_ == 0) & (n_ == 3) )
# Set new values
z_[R1] = 0
z_[R2] = 0
z_[R3] = z_[R3]
z_[R4] = 1
# Make sure borders stay null
z[0,:] = z[-1,:] = z[:,0] = z[:,-1] = 0

birth = (n==3)[1:-1,1:-1] & (z[1:-1,1:-1]==0)
survive = ((n==2) | (n==3))[1:-1,1:-1] & (z[1:-1,1:-1]==1)
z[...] = 0
z[1:-1,1:-1][birth | survive] = 1
