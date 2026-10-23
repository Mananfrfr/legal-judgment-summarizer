import random,unittest
from fast_rouge import overlap_scores,lcs_length
from summary_evaluation import overlap_scores as reference_scores

class FastRougeTests(unittest.TestCase):
    def test_exact_against_reference_random_sequences(self):
        rng=random.Random(42);words=['appeal','allowed','dismissed','court','judge','not','taxes','tax','123']
        for _ in range(100):
            a=' '.join(rng.choices(words,k=rng.randrange(70)))
            b=' '.join(rng.choices(words,k=rng.randrange(70)))
            self.assertEqual(overlap_scores(a,b),reference_scores(a,b))
    def test_empty_repetition_order_and_stemming(self):
        for a,b in [('', ''),('a',''),('a a b','a b b'),('a b c','c b a'),('Courts taxed the claimant.','Court taxes claims!')]:
            self.assertEqual(overlap_scores(a,b),reference_scores(a,b))
        self.assertEqual(lcs_length(['a']*10000,['a']*9000),9000)
