import numpy as np
from scipy import linalg
from utils import *
from mpl_toolkits.mplot3d import Axes3D

def prediction(Mu, Sigma, time):
    U = np.zeros([4, 4])
    U[0, :] = np.array([0, -rotational_velocity[2, i], rotational_velocity[1, i], linear_velocity[0, i]])
    U[1, :] = np.array([rotational_velocity[2, i], 0, -rotational_velocity[0, i], linear_velocity[1, i]])
    U[2, :] = np.array([-rotational_velocity[1, i], rotational_velocity[0, i], 0, linear_velocity[2, i]])
    U[3, :] = np.array([0, 0, 0, 0])
    U = -1 * (time) * U
    U = linalg.expm(U)

    #Covariance

    U_c = np.zeros([6,6])
    U_c[0, : ] = np.array([0, -rotational_velocity[2, i], rotational_velocity[1, i], 0, -linear_velocity[2, i], linear_velocity[1, i]])
    U_c[1, : ] = np.array([rotational_velocity[2, i], 0, -rotational_velocity[0, i], linear_velocity[2, i], 0, -linear_velocity[0, i]])
    U_c[2, : ] = np.array([-rotational_velocity[1, i], rotational_velocity[0, i], 0, -linear_velocity[1, i], linear_velocity[0,i], 0])
    U_c[3, : ] = np.array([0, 0, 0, 0, -rotational_velocity[2, i], rotational_velocity[1, i]])
    U_c[4, : ] = np.array([0, 0, 0, rotational_velocity[2, i], 0, -rotational_velocity[0, i]])
    U_c[5, : ] = np.array([0, 0, 0, -rotational_velocity[1, i], rotational_velocity[0, i], 0])
    # W = [[np.random.normal() for i in range(6)] for j in range(6)]
    # W = np.asanyarray(W)
    W =    10 ** (-7) * np.identity(6)
    #W = 10 ** (-8) * np.random.normal(0,1,size=(6,6))
    return U.dot(Mu), linalg.expm(-1 * time * U_c).dot(Sigma.dot(linalg.expm(-1 * time * U_c).T)) + (time**2) * W

def hat_operator(t):
    h = np.zeros([3,3])
    h[0, :] = np.array([0,    -t[2], t[1]])
    h[1, :] = np.array([t[2],    0,  -t[0]])
    h[2, :] = np.array([-t[1], t[0],  0])
    return h

def Update_landmarks(cam_T_imu, M, m_t, z_t, Sigma, D, Mu):
    t = cam_T_imu.dot(Mu.dot(m_t))
    z_hat = M.dot((t / t[2]))
    # V =   np.identity(4)
    V =   np.identity(4)
    J = np.zeros([4,4])
    J[0,:] = np.array([1, 0, -t[0]/t[2], 0])
    J[1,:] = np.array([0, 1, -t[1]/t[2], 0])
    J[2,:] = np.array([0, 0, 0, 0])
    J[3,:] = np.array([0, 0, -t[3]/ t[2], 1])
    J = J / t[2]

    H = M.dot(J.dot(cam_T_imu.dot(D)))
    # Update Kalman Gain
    K_t = Sigma.dot(H.T.dot(np.linalg.inv(H.dot(Sigma.dot(H.T)) + V)))
    #Update M
    m_t_1 = m_t + D.dot(K_t.dot( z_t - z_hat))
    #Update Sigma
    Sigma = (np.identity(3) - K_t.dot(H)).dot(Sigma)
    return m_t_1, Sigma

def Update_Pose(cam_T_imu, M, m_t, z_t, Sigma, Mu):
    t = cam_T_imu.dot(Mu.dot(m_t))
    z_hat = M.dot((t / t[2]))
    V =    np.identity(4)

    # V =  0.01 * np.random.normal(0, 2, size=(4, 4))
    J = np.zeros([4,4])
    J[0,:] = np.array([1, 0, -t[0]/t[2], 0])
    J[1,:] = np.array([0, 1, -t[1]/t[2], 0])
    J[2,:] = np.array([0, 0, 0, 0])
    J[3,:] = np.array([0, 0, -t[3]/ t[2], 1])
    J = J / t[2]
    #Construct D
    D = np.append(np.block([np.identity(3), -1 * hat_operator(Mu.dot(m_t))]) , np.zeros([1,6]), axis = 0)

    H = M.dot(J.dot(cam_T_imu.dot(D)))

    # Update Kalman Gain
    K_t = Sigma.dot((H.T).dot(np.linalg.inv(H.dot(Sigma.dot(H.T)) + V)))
    #Update M
    # m_t_1 = m_t + D.dot(K_t.dot( z_t - z_hat))
    n = K_t.dot(z_t - z_hat)
    l = n[0:3]
    w = n[3:6]
    U = np.zeros([4, 4])
    U[0, :] = np.array([0, -w[2], w[1], l[0]])
    U[1, :] = np.array([w[2], 0, -w[0], l[1]])
    U[2, :] = np.array([-w[1], w[0], 0, l[2]])
    U[3, :] = np.array([0, 0, 0, 0])
    Mu_new = linalg.expm(U).dot(Mu)
    # Update Sigma
    Sigma_new = (np.identity(6) - K_t.dot(H)).dot(Sigma)
    return Mu_new, Sigma_new

if __name__ == '__main__':
    filename = "./data/0027.npz"
    t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
    Mu = np.identity(4)
    Sigma = 0.01 * np.identity(6)
    t_pre = t[0,0]
    trajectoty = np.zeros([4, 4, t.shape[1]])
    trajectoty[:,:,0] = Mu
    featuresM = np.zeros([4, features.shape[1], features.shape[2]])
    #Create intrinsic Matrix
    M = np.zeros([4,4])
    M[0,:] = np.array([K[0,0], 0, K[0,2], 0])
    M[1,:] = np.array([K[1,0], K[1, 1], K[1, 2], 0])
    M[2,:] = np.array([K[0,0], 0, K[0,2], -K[0,0] * b])
    M[3,:] = np.array([K[1,0], K[1, 1], K[1, 2], 0])

    fub = np.ones([features.shape[1],1]) * K[0,0] * b
    A = np.zeros([4, features.shape[1]])
    A[3,:] = np.ones([features.shape[1], 1]).flatten()
    seen_m = np.zeros([features.shape[1], 1])
    #Dialation matrix
    D = np.append(np.identity(3),np.zeros([1,3]), axis = 0)
    #Covariance for each matrix
    Sigma_M = np.identity(3)

    for i in range(features.shape[1] - 1):
        Sigma_M = np.block([Sigma_M ,  np.identity(3)])
    # Sigma_M = Sigma_M * 0.001
    new_features = np.zeros([4, features.shape[1]])

    #Main loop
    for i in range(0, t.shape[1]):
        t_curr = t[0,i]
        #Prediction
        Mu, Sigma = prediction(Mu, Sigma, t_curr - t_pre)
        trajectoty[:,:, i] = linalg.inv(Mu)
        z =(fub / (features[0, :, i] - features[2, : ,i]).reshape([features.shape[1],1]))
        #A[2, :] = z.flatten()
        x_o_z =  ((features[0, : , i] - K[0, 2]) / K[0, 0]).reshape((features.shape[1],1))
        y_o_z =  ((features[1, :, i] - K[1, 2]) / K[1, 1]).reshape((features.shape[1],1))
        count = 0

        for j in range(A.shape[1]):
            # A[:, j] = np.linalg.inv(Mu).dot((np.linalg.inv(cam_T_imu).dot(A[:, j])))
            if(features[0,j,i]!= -1 and seen_m[j, 0] == 0):
                count = count + 1
                A[0, j] = x_o_z[j, 0] * z[j, 0]
                A[1, j] = y_o_z[j, 0] * z[j, 0]
                A[2, j] = z[j, 0]
                A[:, j] = np.linalg.inv(Mu).dot((np.linalg.inv(cam_T_imu).dot(A[:, j])))
                new_features[:, j] = A[:, j]
                seen_m[j, 0] = 1
                # Mu, Sigma = Update_Pose(cam_T_imu, M, new_features[:, j], features[:, j, i], Sigma, Mu)
            elif(features[0,j,i]!= -1 and seen_m[j, 0] == 1):
                new_features[:, j], Sigma_M[:, j:j + 3] = Update_landmarks(cam_T_imu, M, A[:, j], features[:, j, i], 0.01 * Sigma_M[:, j:j + 3], D, Mu)
                Mu, Sigma =  Update_Pose(cam_T_imu, M, new_features[:, j], features[:, j, i], Sigma, Mu)





        # new_Mu[] = Update_Pose(cam_T_imu, M, m_t, z_t, Sigma, Mu)
        # Update Pose

        '''
        seen_u_m = np.zeros([features.shape[1], 1])
        for j in range(A.shape[1]):
            if(features[0,j,i]!= -1 ):

                # M[2, :] = np.array([0, 0 , 0, K[0,0] * b])
                M = M / z[j,0]
                new_features[:, j], Sigma_M[:,j:j+3] =   Update_landmarks(cam_T_imu, M, A[:,j], features[:,j,i], 0.1 * Sigma_M[:,j:j+3], D)
                #new_features[:, j] =  np.linalg.inv(Mu).dot(new_features[:, j] )

                #def Update_landmarks(cam_T_imu, M, m_t, z_t, Sigma, D):
                seen_u_m[j, 0] = 1
        '''

        #seen_m = 1
        #count
        # for j in range(A.shape[1]):
        #   if(features[0,j,i]!= -1 and seen_m[j, 0] == 0):


        #diag_D = np.kron(np.eye(count),D)
        #l_Sigma = 0.01 * np.identity(count)
        
        #Update_landmarks(cam_T_imu,D, A[:,j], Mu, Sigma)
        #featuresM[:,:,i] = A
        #Update
        #new_feature = Update(cam_T_imu, K, new_features, Mu, Sigma)

        t_pre = t_curr
    #visualize_trajectory_2d(trajectoty, A)
    visualize_trajectory_2d(trajectoty, new_features)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')



    ax.scatter(new_features[0], new_features[1], new_features[2])
    plt.show()

a = 0
    # (a) IMU Localization via EKF Prediction

    # (b) Landmark Mapping via EKF Update

    # (c) Visual-Inertial SLAM (Extra Credit)

    # You can use the function below to visualize the robot pose over time
    #visualize_trajectory_2d(world_T_imu,show_ori=True)
