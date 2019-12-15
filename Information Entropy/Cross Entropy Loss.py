# entropy of examples from a classification task with 3 classes
from math import log2
from numpy import mean
from numpy import asarray
from matplotlib import pyplot


# calculate entropy
def entropy(p):
	return -sum([p[i] * log2(p[i]) for i in range(len(p))])

# calculate cross entropy
def cross_entropy(p, q,ets=1e-15):
	return -sum([p[i]*log2(q[i]+ets) for i in range(len(p))])

# class 1
p = asarray([1,0,0]) + 1e-15
print(round(entropy(p),2))

# class 2
p = asarray([0,1,0]) + 1e-15
print(round(entropy(p),2))

# class 3
p = asarray([0,0,1]) + 1e-15
print(round(entropy(p),2))


# define classification data
p = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]

# calculate cross entropy for each example
results = list()
for i in range(len(p)):
	# create the distribution for each event {0, 1}
	expected = [1.0 - p[i], p[i]]
	predicted = [1.0 - q[i], q[i]]

	# calculate cross entropy for the two events
	ce = cross_entropy(expected, predicted)
	results.append(ce)

	print('>[y=%.1f, yhat=%.1f] ce: %.3f nats' % (p[i], q[i], ce))
    

# calculate the average cross entropy
mean_ce = mean(results)
print('Average Cross Entropy: %.3f nats' % mean_ce)


#Veamos como evoluciona la cross entropia a medida que difieren la distribucion de probabilidad

# define the target distribution for two events
target = [0.0, 1.0]

# define probabilities for the first event
probs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
# create probability distributions for the two events
dists = [[1.0 - p, p] for p in probs]

# calculate cross-entropy for each distribution
ents = [cross_entropy(target, d) for d in dists]

# plot probability distribution vs cross-entropy
pyplot.plot([1-p for p in probs], ents, marker='.')
pyplot.title('Probability Distribution vs Cross-Entropy')
pyplot.xticks([1-p for p in probs], ['[%.1f,%.1f]'%(d[0],d[1]) for d in dists], rotation=70)
pyplot.subplots_adjust(bottom=0.2)
pyplot.xlabel('Probability Distribution')
pyplot.ylabel('Cross-Entropy (nats)')
pyplot.show()