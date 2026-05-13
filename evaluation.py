from util import *

# Add your import statements here
import math


class Evaluation():

	# -------------------------------------------------------------------------
	# Helper: get set of relevant doc IDs for a given query_id from qrels
	# -------------------------------------------------------------------------
	def _get_true_doc_IDs(self, query_id, qrels):
		"""
		Extract relevant document IDs for a given query from the qrels list.
		A document is relevant if its relevance score is between 1 and 4 (inclusive).

		Parameters
		----------
		query_id : int or str
			The query ID
		qrels : list
			List of dicts with keys "query_num", "id", "position"

		Returns
		-------
		set
			Set of relevant document IDs (as integers)
		"""
		true_ids = set()
		for entry in qrels:
			if int(entry["query_num"]) == int(query_id):
				# Relevance is implicitly 1-4; all listed docs are relevant
				true_ids.add(int(entry["id"]))
		return true_ids

	# =========================================================================
	# Precision
	# =========================================================================

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query.

		Precision@k = (# relevant docs in top-k) / k

		Parameters
		----------
		query_doc_IDs_ordered : list
			Ranked list of document IDs returned by the IR system
		query_id : int
			The ID of the query
		true_doc_IDs : set or list
			Ground-truth relevant document IDs for this query
		k : int
			Cut-off rank

		Returns
		-------
		float
			Precision@k value in [0, 1]
		"""
		if k <= 0:
			return 0.0

		top_k = query_doc_IDs_ordered[:k]
		true_set = set(int(d) for d in true_doc_IDs)
		relevant_retrieved = sum(1 for doc_id in top_k if int(doc_id) in true_set)

		precision = relevant_retrieved / k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries.

		Parameters
		----------
		doc_IDs_ordered : list of lists
			Ranked document lists for each query
		query_ids : list
			Query IDs
		qrels : list
			Relevance judgments
		k : int
			Cut-off rank

		Returns
		-------
		float
			Mean Precision@k
		"""
		precisions = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_doc_IDs(query_id, qrels)
			if len(true_doc_IDs) == 0:
				continue  # Skip queries with no relevant docs
			p = self.queryPrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			precisions.append(p)

		meanPrecision = sum(precisions) / len(precisions) if precisions else 0.0
		return meanPrecision

	# =========================================================================
	# Recall
	# =========================================================================

	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query.

		Recall@k = (# relevant docs in top-k) / (total # relevant docs)

		Parameters
		----------
		query_doc_IDs_ordered : list
			Ranked list of document IDs
		query_id : int
			The query ID
		true_doc_IDs : set or list
			Ground-truth relevant document IDs
		k : int
			Cut-off rank

		Returns
		-------
		float
			Recall@k value in [0, 1]
		"""
		true_set = set(int(d) for d in true_doc_IDs)
		if len(true_set) == 0:
			return 0.0

		top_k = query_doc_IDs_ordered[:k]
		relevant_retrieved = sum(1 for doc_id in top_k if int(doc_id) in true_set)

		recall = relevant_retrieved / len(true_set)
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all queries.

		Returns
		-------
		float
			Mean Recall@k
		"""
		recalls = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_doc_IDs(query_id, qrels)
			if len(true_doc_IDs) == 0:
				continue
			r = self.queryRecall(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			recalls.append(r)

		meanRecall = sum(recalls) / len(recalls) if recalls else 0.0
		return meanRecall

	# =========================================================================
	# F0.5-score
	# =========================================================================

	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of F0.5-score of the Information Retrieval System
		at a given value of k for a single query.

		F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
		With beta = 0.5, precision is weighted more than recall.

		Returns
		-------
		float
			F0.5-score@k value in [0, 1]
		"""
		beta = 0.5
		p = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		r = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		denom = (beta ** 2) * p + r
		if denom == 0:
			return 0.0

		fscore = (1 + beta ** 2) * p * r / denom
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of F0.5-score of the Information Retrieval System
		at a given value of k, averaged over all queries.

		Returns
		-------
		float
			Mean F0.5-score@k
		"""
		fscores = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_doc_IDs(query_id, qrels)
			if len(true_doc_IDs) == 0:
				continue
			f = self.queryFscore(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			fscores.append(f)

		meanFscore = sum(fscores) / len(fscores) if fscores else 0.0
		return meanFscore

	# =========================================================================
	# nDCG
	# =========================================================================

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k for a single query.

		DCG@k  = sum_{i=1}^{k} rel_i / log2(i + 1)
		IDCG@k = DCG of the ideal ranking (all relevant docs at the top)
		nDCG@k = DCG@k / IDCG@k

		Relevance is binary here: 1 if doc is relevant, 0 otherwise.

		Returns
		-------
		float
			nDCG@k value in [0, 1]
		"""
		true_set = set(int(d) for d in true_doc_IDs)
		if len(true_set) == 0:
			return 0.0

		# Compute DCG@k
		dcg = 0.0
		for i, doc_id in enumerate(query_doc_IDs_ordered[:k]):
			rel = 1 if int(doc_id) in true_set else 0
			dcg += rel / math.log2(i + 2)  # i+2 because i is 0-indexed (log2(rank+1))

		# Compute ideal DCG@k
		# Ideal: place as many relevant docs as possible in top-k positions
		ideal_rels = [1] * min(len(true_set), k) + [0] * max(0, k - len(true_set))
		idcg = 0.0
		for i, rel in enumerate(ideal_rels):
			idcg += rel / math.log2(i + 2)

		if idcg == 0:
			return 0.0

		nDCG = dcg / idcg
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all queries.

		Returns
		-------
		float
			Mean nDCG@k
		"""
		ndcgs = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_doc_IDs(query_id, qrels)
			if len(true_doc_IDs) == 0:
				continue
			n = self.queryNDCG(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			ndcgs.append(n)

		meanNDCG = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
		return meanNDCG

	# =========================================================================
	# Average Precision (AP)
	# =========================================================================
	def averagePrecision(self, retrieved_doc_IDs, true_doc_IDs):
	    """
	    Computes Average Precision (AP) for a single query
	    using the entire ranked list.
	    """
	
	    num_relevant_found = 0
	    precision_sum = 0.0
	
	    for rank, doc_id in enumerate(retrieved_doc_IDs, start=1):
	
	        if doc_id in true_doc_IDs:
	            num_relevant_found += 1
	
	            precision_at_rank = (
	                num_relevant_found / rank
	            )
	
	            precision_sum += precision_at_rank
	
	    if len(true_doc_IDs) == 0:
	        return 0.0
	
	    return precision_sum / len(true_doc_IDs)


	def meanAveragePrecision(
	    self,
	    doc_IDs_ordered,
	    query_ids,
	    q_rels
	):
	    """
	    Computes true MAP over all queries
	    using the full ranked retrieval list.
	    """
	
	    average_precisions = []
	
	    for i, query_id in enumerate(query_ids):
	
	        true_doc_IDs = self._get_true_doc_IDs(
	            query_id,
	            q_rels
	        )
	
	        if len(true_doc_IDs) == 0:
	            continue
	
	        ap = self.averagePrecision(
	            doc_IDs_ordered[i],
	            true_doc_IDs
	        )
	
	        average_precisions.append(ap)
	
	    if len(average_precisions) == 0:
	        return 0.0
	
	    return sum(average_precisions) / len(average_precisions)

	# =========================================================================
# Reciprocal Rank (RR) and MRR
# =========================================================================

	# =========================================================================
# Reciprocal Rank (RR) and MRR
# =========================================================================
	
	def queryReciprocalRank(
	    self,
	    query_doc_IDs_ordered,
	    true_doc_IDs
	):
	    """
	    Computes Reciprocal Rank (RR) for a single query
	    using the full ranked list.
	
	    RR = 1 / rank of first relevant document
	    """
	
	    true_set = set(int(d) for d in true_doc_IDs)
	
	    for rank, doc_id in enumerate(
	        query_doc_IDs_ordered,
	        start=1
	    ):
	
	        if int(doc_id) in true_set:
	            return 1.0 / rank
	
	    return 0.0
	
	
	def meanReciprocalRank(
	    self,
	    doc_IDs_ordered,
	    query_ids,
	    qrels
	):
	    """
	    Computes true Mean Reciprocal Rank (MRR)
	    over all queries using full rankings.
	    """
	
	    reciprocal_ranks = []
	
	    for i, query_id in enumerate(query_ids):
	
	        true_doc_IDs = self._get_true_doc_IDs(
	            query_id,
	            qrels
	        )
	
	        if len(true_doc_IDs) == 0:
	            continue
	
	        rr = self.queryReciprocalRank(
	            doc_IDs_ordered[i],
	            true_doc_IDs
	        )
	
	        reciprocal_ranks.append(rr)
	
	    if len(reciprocal_ranks) == 0:
	        return 0.0
	
	    return (
	        sum(reciprocal_ranks)
	        / len(reciprocal_ranks)
	    )
