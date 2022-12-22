from __future__ import print_function
from __future__ import division
#from __future__ import unicode_literals
import networkx as nx
import pandas as pd

class PPrDiffuser:

	def __init__(self, network, lower_w, upper_w, max_penalty, edge_scores = None, edge_stress = None, edge_boost = None):
		'''
			PPrDiffuser: object to perform the Personalized PageRank Algorithm
			This method creates the diffuser object from an networkx DiGraph() object,
			which can then be used to diffuse vectors over this

			Input:
				- network : a network hash object
		'''

		# create a directed graph with NetworkX
		self.G = nx.DiGraph()
		# create a reversed graph for diffusion from the 'target' set
		self.G_reversed = nx.DiGraph()
		self.lower_w = lower_w
		self.upper_w = upper_w
		self.max_penalty = max_penalty
		self.edge_boost = edge_boost
		self.edge_stress = edge_stress
		# convert network format to networkX Graph object
		for source in network:
			for (i,t) in network[source]:
				self.G.add_edge(source, t, weight = 1)
				self.G_reversed.add_edge(t, source, weight = 1)

		if (edge_scores is not None):
			self.add_edge_weights(edge_scores)


	def weight_helper(self, edge_score):
		## Following a linear function, cutoff input at the given extremes
		x = max(self.lower_w, edge_score)
		x = min(self.upper_w, x)

		## This penalizes the weight for low scoring edges
		## Max weight is 1, smallest is max_penalty. Default weight is also 1
		## Gives weight 1 edge scores above upper_w.
		edge_weight = (1 - self.max_penalty) + ((x - self.lower_w)/(self.upper_w - self.lower_w)) * self.max_penalty
		return edge_weight
	
	def add_edge_weights(self, edge_scores):
		net_w = pd.read_csv(edge_scores, sep = "\t", header = None)

		for i in range(len(net_w.index)):
			#network[net_w.iloc[i, 1]][net_w.iloc[i, ]]['edge_weigth'] = weight_helper(net_w[4][i])
			self.G[net_w[0][i]][net_w[2][i]]['weight'] = self.weight_helper(net_w[3][i])
			self.G_reversed[net_w[2][i]][net_w[0][i]]['weight'] = self.weight_helper(net_w[3][i])
		
		if self.edge_stress is not None:
			net_s = pd.read_csv(self.edge_stress, sep = "\t", header = None)
			# net_s[3] = net_s[3]**2
			max_count = max(net_s[3].to_list())
			print(max_count)
			if self.edge_boost is None:
				self.edge_boost = (1 - self.max_penalty)**(-1/2)
			for i in range(len(net_s.index)):
				self.G[net_s[0][i]][net_s[2][i]]['weight'] = min(self.G[net_s[0][i]][net_s[2][i]]['weight'] * (net_s[3][i]/max_count) * self.edge_boost, 1)
				self.G_reversed[net_s[2][i]][net_s[0][i]]['weight'] = min(self.G_reversed[net_s[2][i]][net_s[0][i]]['weight'] * (net_s[3][i]/max_count) * self.edge_boost, 1)

		sanity_check = nx.to_pandas_edgelist(self.G)
		# sanity_check.to_csv('edge_weights_sanity_check.txt', sep = "\t", index = False)


	def personal_page_rank(self, p_vector, reverse=False):
		'''
			Personal_Page_Rank: Get the personal pagerank of the supplied input vector

			Input:
				- p_vector: A hash-map of input values for a selection (or all) nodes
				(if supplied nodes aren't in the graph, they will be ignored)

			Output:
				- A vector of diffused heats in hash-map (key,value) format
		'''
		input_pvec = None
		#  without initializing this vector the initial probabilities will be flat
		# and this will be equivalent to standard page rank
		if p_vector:
			input_pvec = {}
			# doesn't seem to be necessary for a non-zero epsilon now, but
			# leave this as a place holder
			epsilon = 0.0
			for node in self.G.nodes(data=False):
				if node in p_vector:
					input_pvec[node] = p_vector[node]
				else:
					input_pvec[node] = epsilon

		if reverse:
			return nx.pagerank_numpy(self.G_reversed, 0.85, input_pvec)
		else:
			return nx.pagerank_numpy(self.G, 0.85, input_pvec)

	def diffuse(self, p_vector, reverse=False):
		'''
			Diffuse: perform generalized diffusion from the supplied input vector

			Input:
				- p_vector: A hash-map of input values for a selection (or all) nodes
				(if supplied nodes aren't in the graph, they will be ignored)

			Output:
				- A vector of diffused heats in hash-map (key,value) format
		'''
		return self.personal_page_rank(p_vector, reverse)
