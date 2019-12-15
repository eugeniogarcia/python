import math as math

# define distributions
events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]

# plot of distributions
from matplotlib import pyplot

# define distributions
events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]

# plot first distribution
pyplot.subplot(2,1,1)
pyplot.bar(events, p)

# plot second distribution
pyplot.subplot(2,1,2)
pyplot.bar(events, q)

# show the plot
pyplot.show()


# calculate cross entropy
def cross_entropy(p, q):
  return -sum([p[i]*math.log2(q[i]) for i in range(len(p))])

# calculate the kl divergence KL(P || Q)
def kl_divergence(p, q):
	return sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))

# calculate entropy H(P)
def entropy(p):
	return -sum([p[i] * math.log2(p[i]) for i in range(len(p))])

# calculate cross entropy H(P, Q)
# metodo alternativo a cross_entropy
def cross_entropy_alt(p, q):
	return entropy(p) + kl_divergence(p, q)


# calculate H(P)
en_p = entropy(p)
print('H(P): %.3f bits' % en_p)

# calculate cross entropy H(Q, Q)
ce_pp = cross_entropy(p, p)
print('H(P, P): %.3f bits' % ce_pp)

# calculate H(Q)
en_q = entropy(q)
print('H(Q): %.3f bits' % en_q)

# calculate cross entropy H(Q, Q)
ce_qq = cross_entropy(q, q)
print('H(Q, Q): %.3f bits' % ce_qq)

# calculate cross entropy H(P, Q)
ce_pq = cross_entropy(p, q)
print('H(P, Q): %.3f bits' % ce_pq)

# calculate cross entropy H(Q, P)
ce_qp = cross_entropy(q, p)
print('H(Q, P): %.3f bits' % ce_qp)

# calculate kl divergence KL(P || Q)
kl_pq = kl_divergence(p, q)
print('KL(P || Q): %.3f bits' % kl_pq)

# calculate cross entropy H(Q, P)
kl_qp = kl_divergence(q, p)
print('KL(Q || P): %.3f bits' % kl_qp)
