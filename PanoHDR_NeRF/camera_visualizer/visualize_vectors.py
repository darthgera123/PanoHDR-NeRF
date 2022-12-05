import numpy as np
import matplotlib.pyplot as plt
from ai import cs

W = 64
H = 32

u, v = np.meshgrid(np.arange(W), np.arange(H))
norm_mat = np.array([[1, 0, -0.5*(W-1)], [0, 1, -0.5*(H-1)], [0, 0, max(H, W)]])
u = u.reshape(-1).astype(dtype=np.float32)
v = v.reshape(-1).astype(dtype=np.float32)
pixels = np.stack((u, v, np.ones_like(u)), axis=0)
pixels = np.dot(norm_mat, pixels)
pixels = pixels / pixels[2, :]
lon = pixels[0, :] * 2 * np.pi
lat = -pixels[1, :] * 2 * np.pi
x = np.cos(lat) * np.sin(lon)
y = -np.sin(lat)
z = np.cos(lat) * np.cos(lon)
rays_d = np.row_stack([x, y, z])
# rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
rays_d = rays_d.transpose((1, 0))  # (H*W, 3)


# rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
#
#



# phi = np.linspace(-np.pi, np.pi, W)
# theta = np.linspace(-np.pi / 2, np.pi / 2, H)
# x_indx = range(W)
# y_indx = range(H)
# phi_array, theta_array = np.array(np.meshgrid(phi, theta))
# x_indx_array, y_indx_array = np.array(np.meshgrid(x_indx, y_indx))
# x_rel, y_rel, z_rel = cs.sp2cart(r=np.ones(theta_array.size), phi=phi_array.reshape(-1), theta=theta_array.reshape(-1))
# vectors = np.stack((x_rel, y_rel, z_rel), axis=0).astype(dtype=np.float32)
# print("vectors.shape")
# print(vectors.shape)
# rays_d = vectors.transpose((1, 0))  # (H*W, 3)
# print("rays_d.size")
# print(vectors.dtype)
#
# for i in range(x_rel.reshape(-1).size):
#     # if x_rel.reshape(-1)[i] > 0:
#     index_string = str(x_indx_array.reshape(-1)[i]) + " , " + str(y_indx_array.reshape(-1)[i])
#     polar_string = str(theta_array.reshape(-1)[i] / np.pi) + " , " + str(phi_array.reshape(-1)[i] / np.pi)
#     print("=======================")
#     print(index_string)
#     print(polar_string)
#
# signs_x = np.sign(vectors[0, :])
# signs_y = np.sign(vectors[1, :])
# signs_z = np.sign(vectors[2, :])
# colors_x = np.zeros((H, W, 3))
# colors_y = np.zeros((H, W, 3))
# colors_z = np.zeros((H, W, 3))
#
# x_pos = []
# x_neg = []
# y_pos = []
# y_neg = []
# z_pos = []
# z_neg = []
#
# for i in range(W * H):
#     if signs_x[i] > 0:
#         colors_x[j, i, :] = [0, 255, 0]
#         x_pos.append(vectors[:, i])
#     else:
#         colors_x[j, i, :] = [255, 0, 0]
#         x_neg.append(vectors[:, i])
#
#     if signs_y[i] > 0:
#         colors_y[j, i, :] = [0, 255, 0]
#         y_pos.append(vectors[:, i])
#     else:
#         colors_y[j, i, :] = [255, 0, 0]
#         y_neg.append(vectors[:, i])
#
#     if signs_z[i] > 0:
#         colors_z[j, i, :] = [0, 255, 0]
#         z_pos.append(vectors[:, i])
#     else:
#         colors_z[j, i, :] = [255, 0, 0]
#         z_neg.append(vectors[:, i])
#
#
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 3, 1)
# imgplot = plt.imshow(colors_x.astype(np.uint8))
# ax.set_title('X')
# ax = fig.add_subplot(1, 3, 2)
# imgplot = plt.imshow(colors_y.astype(np.uint8))
# ax.set_title('Y')
# ax = fig.add_subplot(1, 3, 3)
# imgplot = plt.imshow(colors_z.astype(np.uint8))
# ax.set_title('Z')
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 3, 1, projection='3d')
# test_pos_x = [pos[0] for pos in x_pos]
# test_pos_y = [pos[1] for pos in x_pos]
# test_pos_z = [pos[2] for pos in x_pos]
# ax.plot(test_pos_x, test_pos_y, test_pos_z, 'go')
# test_neg_x = [pos[0] for pos in x_neg]
# test_neg_y = [pos[1] for pos in x_neg]
# test_neg_z = [pos[2] for pos in x_neg]
# ax.plot(test_neg_x, test_neg_y, test_neg_z, 'ro')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('X Value')
# ax.legend()
# #
# ax = fig.add_subplot(1, 3, 2, projection='3d')
# # ax = fig.add_subplot(projection='3d')
# # for i in range(x_rel.reshape(-1).size):
# #     # if x_rel.reshape(-1)[i] > 0:
# #     string = str(x_indx_array.reshape(-1)[i]) + "," + str(y_indx_array.reshape(-1)[i])
# #     ax.text(x_rel.reshape(-1)[i], y_rel.reshape(-1)[i], z_rel.reshape(-1)[i], string, color='black')
# test_pos_x = [pos[0] for pos in y_pos]
# test_pos_y = [pos[1] for pos in y_pos]
# test_pos_z = [pos[2] for pos in y_pos]
# ax.plot(test_pos_x, test_pos_y, test_pos_z, 'go')
# test_neg_x = [pos[0] for pos in y_neg]
# test_neg_y = [pos[1] for pos in y_neg]
# test_neg_z = [pos[2] for pos in y_neg]
# ax.plot(test_neg_x, test_neg_y, test_neg_z, 'ro')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Y Value')
# ax.legend()
#
# ax = fig.add_subplot(1, 3, 3, projection='3d')
# test_pos_x = [pos[0] for pos in z_pos]
# test_pos_y = [pos[1] for pos in z_pos]
# test_pos_z = [pos[2] for pos in z_pos]
# ax.plot(test_pos_x, test_pos_y, test_pos_z, 'go')
# test_neg_x = [pos[0] for pos in z_neg]
# test_neg_y = [pos[1] for pos in z_neg]
# test_neg_z = [pos[2] for pos in z_neg]
# ax.plot(test_neg_x, test_neg_y, test_neg_z, 'ro')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Z Value')
# ax.legend()
#
# plt.show()