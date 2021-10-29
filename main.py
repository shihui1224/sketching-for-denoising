
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sketch.sketch_wrapper import SketchWrapper, load_sk_options
from estimator.gmm_estimator import GmmEstimator, load_estimator_options
from tools import generator

if __name__ == '__main__':
    """Number of samples per component"""
    n_samples = 10000
    """Generate random sample, k components, d dimensions"""
    # np.random.seed(1000)
    k = 4
    d = 2
    truegmm = generator.generate_gmm(d, k)
    X = generator.draw_gmm(truegmm, n_samples)
    cmap = plt.cm.get_cmap('tab20', k)

    """plot"""
    generator.plot_results(X, truegmm.mu, truegmm.Sigma, cmap, 'True GMM')

    """EM learn"""
    X_em = X.transpose()
    gmm_em = GaussianMixture(n_components=k, means_init=np.zeros([k, d]), random_state=0).fit(X_em)
    labels = gmm_em.predict(X_em)
    plt.figure(2)
    generator.plot_results(X, gmm_em.means_, gmm_em.covariances_, cmap, 'Estimation by EM')

    """sketch parameters"""
    m = 500
    r = 2
    sk_options = load_sk_options()
    """set sketch"""
    skw = SketchWrapper(X, m, sk_options)
    skw.set_sketch(X)
    est_options = load_estimator_options()
    """estimating model"""
    estmix = GmmEstimator(skw, est_options, r)
    nres = estmix.estimate(k)
    plt.figure(3)
    generator.plot_results(X, estmix.mixture.mu, estmix.mixture.Sigma, cmap, 'Estimation by sketching')
    plt.figure(4)
    plt.plot(nres)
    plt.title('Energie')
    plt.show()




