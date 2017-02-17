from pypac import concepts, learner, tools


n = 16
instance = concepts.GaussianLtf(n)
print('generated Gaussian LTF instance with %s variables' % n)

k = 4
epsilon = .05
delta = .05
learner = learner.LowDegreeAlgorithm(instance, k, epsilon, delta)
print('learning with low degree algorithm for k<=%s, epsilon=%s, delta=%s ...' % (k, epsilon, delta))

model = learner.learn()
print('learning completed, checking accuracy ...')

sample_size = 1000
dist = tools.approx_dist(instance, model, sample_size)
print('average distance on %s random sample inputs: %s' % (sample_size, dist))
