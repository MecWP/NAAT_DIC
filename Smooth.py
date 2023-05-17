import numpy as np #check
import scipy.ndimage as ndimage #check
import cv2 #check

class SmoothMixin:
    def SmoothT(self,I,Filter,Value):

        if self.CB_cluster.IsChecked() == True:
            from sklearn.cluster import KMeans
            n_components = int(self.ET_clusterNumber.GetLineText(0))
            gmm = KMeans(n_clusters=n_components, random_state=0)#mixture.GaussianMixture(n_components=n_components)
            X = np.squeeze(np.array([self.X.flatten().reshape((1,-1)),self.Y.flatten().reshape((1,-1)),I.flatten().reshape((1,-1))]))
            X = X.reshape((-1,3))

            for idx in range(3):
                X[np.isnan(X)[:,idx],:] = 0

            # Limit X
            Variables = int(self.ET_trainingVector.GetLineText(0))

            if Variables == 1:
                pass
            elif Variables == 2:
                X = X[:,2].reshape((-1,1))
            elif Variables == 3:
                X = X[:,[0,1]]
            #eval('X = X[:,'+Variables+']')



            gmm.fit(X)
            Index = gmm.predict(X)

            I_all = []
            for idx in range(n_components):
                I_temp = I.flatten()
                I_temp[~(Index==idx)] = np.nan

                I_all.append(I_temp.reshape(I.shape))


        if Filter == 0:


            if self.CB_cluster.IsChecked() == True:

                Smooth = np.zeros_like(I)
                for idx in range(n_components):

                    U = I_all[idx].copy()
                    U[U != I_all[idx].copy()] = 0
                    UU = ndimage.gaussian_filter(U, sigma=(Value, Value), order=0)  #

                    V = I_all[idx].copy() * 0 + 1
                    V[U != I_all[idx].copy()] = 0
                    VV = ndimage.gaussian_filter(V, sigma=(Value, Value), order=0)  #

                    Smooth_temp = UU / VV
                    Smooth_temp[np.isnan(Smooth_temp)] = 0
                    Smooth += Smooth_temp


            else:
            #Smooth =  ndimage.gaussian_filter(I, sigma=(Value, Value), order=0)
                U = I.copy()
                U[U != I.copy()] = 0
                UU = ndimage.gaussian_filter(U, sigma=(Value, Value), order=0)#

                V = I.copy() * 0 + 1
                V[U != I.copy()] = 0
                VV = ndimage.gaussian_filter(V, sigma=(Value, Value), order=0)#

                Smooth = UU / VV

                Filter_length = float(self.ET_filterSizeDisp.GetLineText(0))

                if Filter_length>0: # In case use both filters
                    U = Smooth.copy()
                    U[U != Smooth.copy()] = 0
                    UU = cv2.bilateralFilter(U.astype(np.float32), int(Filter_length), int(Filter_length / 2),
                                             int(Filter_length / 2))  # ndimage.filters.median_filter(U, size=int(Value))

                    V = Smooth.copy() * 0 + 1
                    V[U != Smooth.copy()] = 0
                    VV = cv2.bilateralFilter(V.astype(np.float32), int(Filter_length), int(Filter_length / 2),
                                             int(Filter_length / 2))  ## ndimage.filters.median_filter(V, size=int(Value))

                    Smooth = UU / VV


        if Filter == 1:

            #Smooth = ndimage.gaussian_filter(I, sigma=(Value, Value), order=0)
            U = I.copy()
            U[U != I.copy()] = 0
            UU = cv2.bilateralFilter(U.astype(np.float32),int(Value),int(Value/2),int(Value/2))#ndimage.filters.median_filter(U, size=int(Value))

            V = I.copy() * 0 + 1
            V[U != I.copy()] = 0
            VV = cv2.bilateralFilter(V.astype(np.float32),int(Value),int(Value/2),int(Value/2))## ndimage.filters.median_filter(V, size=int(Value))

            Smooth = UU / VV

            Std = float(self.ET_stdGausDisp.GetLineText(0))

            if Std > 0:  # In case use both filters

                U = Smooth.copy()
                U[U != Smooth.copy()] = 0
                UU = ndimage.gaussian_filter(U, sigma=(Std, Std), order=0)  #

                V = Smooth.copy() * 0 + 1
                V[U != Smooth.copy()] = 0
                VV = ndimage.gaussian_filter(V, sigma=(Std, Std), order=0)  #

                Smooth = UU / VV

        return Smooth