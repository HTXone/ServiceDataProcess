import numpy as np
from collections import Counter, OrderedDict

class RGN:
    '''Rank Gaussian Normalization'''
    def __init__(self, data=None, precision=np.float32):
        #data: 1D array or list
        self._data = data
        self.precision = precision
        self._output = None
        if self._data is None:
            self._trafo_map = None
        else:
            self.fit_transform(self._data)

    @property
    def data(self):
        return self._data

    @property
    def output(self):
        return self._output

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, p):
        if not isinstance(p, type):
            raise ValueError('precision must be a data type, e.g.: np.float64')
        self._precision = p

    def _RationalApproximation(self, t:float)->float:
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]
        return t - ((c[2]*t + c[1])*t + c[0]) / (((d[2]*t + d[1])*t + d[0])*t + 1.0)

    def _NormalCDFInverse(self, p:float) -> float:

        if (p <= 0.0 or p >= 1.0):
            raise Exception('0<p<1. The value of p was: {}'.format(p))
        if (p < 0.5):
            return -self._RationalApproximation(np.sqrt(-2.0*np.log(p)) )
        return self._RationalApproximation( np.sqrt(-2.0*np.log(1-p)) )

    def _vdErfInvSingle01(self, x:float) -> float:
        if x == 0:
            return 0
        elif x < 0:
            return -self._NormalCDFInverse(-x)*0.7
        else:
            return self._NormalCDFInverse(x)*0.7

    def fit_transform(self, dataIn:list) -> dict:
        self.fit(dataIn)
        return self.transform(dataIn)

    def fit(self, dataIn:list):
        self._data = dataIn
        trafoMap = OrderedDict()
        hist = Counter(dataIn)
        if len(hist) == 0:
            pass
        elif len(hist) == 1:
            key = list(hist.keys())[0]
            trafoMap[key] = 0.0
        elif len(hist) == 2:
            keys = sorted(list(hist.keys()))
            trafoMap[keys[0]] = 0.0
            trafoMap[keys[1]] = 1.0
        else:
            N = cnt = 0
            for it in hist:
                N += hist[it]
            assert (N == len(dataIn))
            mean = 0.0
            for it in sorted(list(hist.keys())):
                rankV = cnt / N
                rankV = rankV * 0.998 + 1e-3
                rankV = self._vdErfInvSingle01(rankV)
                assert(rankV >= -3.0 and rankV <= 3.0)
                mean += hist[it] * rankV
                trafoMap[it] = rankV
                cnt += hist[it]
            mean /= N
            for it in trafoMap:
                trafoMap[it] -= mean
        self._trafo_map = trafoMap
        return

    def _binary_search(self, keys, val):
        start, end = 0, len(keys)-1
        while start+1 < end:
            mid = (start + end) // 2
            if val < keys[mid]:
                end = mid
            else:
                start = mid
        return keys[start], keys[end]

    def transform(self, dataIn:list) -> dict:
        dataOut = []
        trafoMap = self._trafo_map
        keys = list(trafoMap.keys())
        if len(keys) == 0:
            raise Exception('No transfermation map')
        for i in range(len(dataIn)):
            val = dataIn[i]
            trafoVal = 0.0
            if val <= keys[0]:
                trafoVal = trafoMap[keys[0]]
            elif val >= keys[-1]:
                trafoVal = trafoMap[keys[-1]]
            elif val in trafoMap:
                trafoVal = trafoMap[val]
            else:
                lower_key, upper_key = self._binary_search(keys, val)
                x1, y1 = lower_key, trafoMap[lower_key]
                x2, y2 = upper_key, trafoMap[upper_key]

                trafoVal = y1 + (val - x1) * (y2 - y1) / (x2 - x1)
            dataOut.append(trafoVal)
        dataOut = np.asarray(dataOut, dtype=self.precision)
        self._output = dataOut
        return self._output



from sklearn.linear_model import LogisticRegression

class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        #权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        #使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        #训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        #训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        #权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                #L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    #对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    #计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self

if __name__ == '__main__':
    data = [-19.9378,10.5341,-32.4515,33.0969,24.3530,-1.1830,-1.4106,-4.9431,
        14.2153,26.3700,-7.6760,60.3346,36.2992,-126.8806,14.2488,-5.0821,
        1.6958,-21.2168,-49.1075,-8.3084,-1.5748,3.7900,-2.1561,4.0756,
        -9.0289,-13.9533,-9.8466,79.5876,-13.3332,-111.9568,-24.2531,120.1174]
    rgn = RGN(data)
    print(rgn.output)