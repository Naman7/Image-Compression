import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Cr7.jpg",0)

a = np.asarray(img)

q = int(input("Quality Level in % \n"))
u,s,v = np.linalg.svd(a, full_matrices=0)

q = round((q * s.shape[0])/100)


So = u.shape[0]*u.shape[1] + v.shape[0]*v.shape[1] + s.shape[0]**2

u=u[:, 0:q]
v=v[0:q]
x = np.zeros((q, q), float)

for i in range(q):
    x[i][i] = s[i]

z = np.matmul(u, x)
z = np.matmul(z, v)

org = (s.shape[0])**2 * (u.shape[0] + v.shape[1])
now = q**2 * (a.shape[0] + q)

Sn = u.shape[0]*q + q*v.shape[1] + q**2

print("Reduction in computation is ", round(((org-now)/org)*100), "%")
print("Reduction in space is", round(((So-Sn)/So)*100), "%")

plt.imshow(z, cmap='gray')

